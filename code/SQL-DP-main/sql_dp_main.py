from __future__ import absolute_import, division, print_function

import os
import logging
from timeit import default_timer
import numpy as np
import tensorflow as tf
import tensorflow_fold as td

import apputil
import data
import embedding
from config import hyper, param

import pandas as pd
import json
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


logger = logging.getLogger(__name__)


def identity_initializer():
    def _initializer(shape, dtype=np.float32):
        if len(shape) == 1:
            return tf.constant(1., dtype=dtype, shape=shape)
        elif len(shape) == 2 and shape[0] == shape[1]:
            return tf.constant(np.identity(shape[0], dtype))
        elif len(shape) == 4 and shape[2] == shape[3]:
            array = np.zeros(shape, dtype=float)
            cx, cy = shape[0]/2, shape[1]/2
            for i in range(shape[2]):
                array[cx, cy, i, i] = 1
            return tf.constant(array, dtype=dtype)
        else:
            raise
    return _initializer


# get coding layer
def coding_blk():
    """Input: node dict
    Output: TensorType([1, hyper.word_dim])
    """
    Wcomb1 = param.get('Wcomb1')
    Wcomb2 = param.get('Wcomb2')

    blk = td.Composition()
    with blk.scope():
        direct = embedding.direct_embed_blk().reads(blk.input)
        composed = embedding.composed_embed_blk().reads(blk.input)
        Wcomb1 = td.FromTensor(param.get('Wcomb1'))
        Wcomb2 = td.FromTensor(param.get('Wcomb2'))

        direct = td.Function(embedding.batch_mul).reads(direct, Wcomb1)
        composed = td.Function(embedding.batch_mul).reads(composed, Wcomb2)

        added = td.Function(tf.add).reads(direct, composed)
        blk.output.reads(added)
    return blk


def collect_node_for_conv_patch_blk(max_depth=2):
    """Input: node dict
    Output: flattened list of all collected nodes, in the format
    [(node, idx, pclen, depth, max_depth), ...]
    """
    def _collect_patch(node):
        collected = [(node, 1, 1, 0, max_depth)]

        def recurse_helper(node, depth):
            if depth > max_depth:
                return
            for idx, c in enumerate(node['children'], 1):
                collected.append((c, idx, node['clen'], depth + 1, max_depth))
                recurse_helper(c, depth + 1)

        recurse_helper(node, 0)
        return collected

    return td.InputTransform(_collect_patch)


def tri_combined(idx, pclen, depth, max_depth):
    """TF function, input: idx, pclen, depth, max_depth as batch (1D Tensor)
    Output: weight tensor (3D Tensor), first dim is batch
    """
    Wconvt = param.get('Wconvt')
    Wconvl = param.get('Wconvl')
    Wconvr = param.get('Wconvr')

    dim = tf.unstack(tf.shape(Wconvt))[0]
    batch_shape = tf.shape(idx)

    tmp = (idx - 1) / (pclen - 1)
    # when pclen == 1, replace nan items with 0.5
    tmp = tf.where(tf.is_nan(tmp), tf.ones_like(tmp) * 0.5, tmp)

    t = (max_depth - depth) / max_depth
    r = (1 - t) * tmp
    l = (1 - t) * (1 - r)

    lb = tf.transpose(tf.transpose(tf.eye(dim, batch_shape=batch_shape)) * l)
    rb = tf.transpose(tf.transpose(tf.eye(dim, batch_shape=batch_shape)) * r)
    tb = tf.transpose(tf.transpose(tf.eye(dim, batch_shape=batch_shape)) * t)

    lb = tf.reshape(lb, [-1, dim])
    rb = tf.reshape(rb, [-1, dim])
    tb = tf.reshape(tb, [-1, dim])

    tmp = tf.matmul(lb, Wconvl) + tf.matmul(rb, Wconvr) + tf.matmul(tb, Wconvt)

    tmp = tf.reshape(tmp, [-1, hyper.word_dim, hyper.conv_dim])
    return tmp


def tri_combined_blk():
    blk = td.Function(tri_combined, infer_output_type=False)
    blk.set_output_type(td.TensorType([hyper.word_dim, hyper.conv_dim]))
    return blk


def weighted_feature_blk():
    """Input: (feature                       , idx   , pclen,  depth,  max_depth)
              (TensorType([hyper.word_dim, ]), Scalar, Scalar, Scalar, Scalar)
    Output: weighted_feature
            TensorType([hyper.conv_dim, ])
    """
    blk = td.Composition()
    with blk.scope():
        fea = blk.input[0]
        Wi = tri_combined_blk().reads(blk.input[1], blk.input[2], blk.input[3], blk.input[4])

        weighted_fea = td.Function(embedding.batch_mul).reads(fea, Wi)

        blk.output.reads(weighted_fea)
    return blk


def feature_detector_blk(max_depth=2):
    """Input: node dict
    Output: TensorType([hyper.conv_dim, ])
    Single patch of the conv. Depth is max_depth
    """
    blk = td.Composition()
    with blk.scope():
        nodes_in_patch = collect_node_for_conv_patch_blk(max_depth=max_depth).reads(blk.input)

        mapped = td.Map(td.Record((coding_blk(), td.Scalar(), td.Scalar(), td.Scalar(), td.Scalar()))).reads(nodes_in_patch)

        # compute weighted feature for each elem
        weighted = td.Map(weighted_feature_blk()).reads(mapped)

        # add together
        added = td.Reduce(td.Function(tf.add)).reads(weighted)

        # add bias
        biased = td.Function(tf.add).reads(added, td.FromTensor(param.get('Bconv')))

        # tanh
        tanh = td.Function(tf.nn.tanh).reads(biased)

        blk.output.reads(tanh)
    return blk


# generalize to tree_fold, accepts one block that takes two node, returns a value
def dynamic_pooling_blk():
    """Input: root node dic
    Output: pooled, TensorType([hyper.conv_dim, ])
    """
    leaf_case = feature_detector_blk()

    pool_fwd = td.ForwardDeclaration(td.PyObjectType(), td.TensorType([hyper.conv_dim, ]))
    pool = td.Composition()
    with pool.scope():
        cur_fea = feature_detector_blk().reads(pool.input)
        children = td.GetItem('children').reads(pool.input)

        mapped = td.Map(pool_fwd()).reads(children)
        summed = td.Reduce(td.Function(tf.maximum)).reads(mapped)
        summed = td.Function(tf.maximum).reads(summed, cur_fea)
        pool.output.reads(summed)
    pool = td.OneOf(lambda x: x['clen'] == 0,
                    {True: leaf_case, False: pool})
    pool_fwd.resolve_to(pool)
    return pool


def build_model():
    # create model variables
    param.initialize_tbcnn_weights()

    # Compile the block and append fc layers
    tree_pooling = dynamic_pooling_blk()
    # td is tensorflow fold
    compiler = td.Compiler.create((tree_pooling, td.Scalar(dtype='float32')))
    (pooled, batched_labels) = compiler.output_tensors  # batched_labels shape=(?,), dtype=float32

    fc1 = tf.nn.relu(tf.add(tf.matmul(pooled, param.get('FC1/weight')), param.get('FC1/bias')))
    fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, param.get('FC2/weight')), param.get('FC2/bias')))

    difficulty = tf.nn.sigmoid(fc2)  # Tensor("Sigmoid:0", shape=(?, 1), dtype=float32)
    loss_mse = tf.reduce_mean(tf.square(batched_labels - difficulty))
    batch_size_op = tf.unstack(tf.shape(batched_labels))[0]  # Tensor("unstack:0", shape=(), dtype=int32)
    return compiler, fc1, fc2, difficulty, batched_labels, loss_mse, batch_size_op, batched_labels


# get the code structure feature
def tbcnn_svm():
    fc1_all, y_label = [], []
    ds = data.load_dataset('../data/statements')
    hyper.node_type_num = len(ds.word2int)  # node_type_number
    (compiler, fc1, _, _, _, raw_mse, batch_size_op, batched_label) = build_model()

    # restorer for embedding matrix
    embedding_path = tf.train.latest_checkpoint(hyper.embedding_dir)
    if embedding_path is None:
        raise ValueError('Path to embedding checkpoint is incorrect: ' + hyper.embedding_dir)

    # restorer for other variables
    checkpoint_path = tf.train.latest_checkpoint(hyper.train_dir)
    if checkpoint_path is None:
        raise ValueError('Path to tbcnn checkpoint is incorrect: ' + hyper.train_dir)

    restored_vars = tf.get_collection_ref('restored')
    restored_vars.append(param.get('We'))
    restored_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    embeddingRestorer = tf.train.Saver()
    restorer = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    # train loop
    total_size, test_gen = ds.get_split('test')
    test_set = compiler.build_loom_inputs(test_gen)

    with tf.Session() as sess:
        # Restore embedding matrix first
        embeddingRestorer.restore(sess, embedding_path)
        # Restore others
        restorer.restore(sess, checkpoint_path)
        # Initialize other variables
        gvariables = [v for v in tf.global_variables() if v not in tf.get_collection('restored')]
        sess.run(tf.variables_initializer(gvariables))

        num_epochs = 1 if not hyper.warm_up else 3
        for shuffled in td.epochs(test_set, num_epochs):
            logger.info('')
            logger.info('======================= TBCNN ====================================')
            accumulated_mse = 0.
            start_time = default_timer()
            for step, batch in enumerate(td.group_by_batches(shuffled, hyper.batch_size), 1):
                feed_dict = {compiler.loom_input_tensor: batch}
                fc_one, mse_value, actual_bsize, label = sess.run([fc1, raw_mse, batch_size_op, batched_label], feed_dict)

                for k in fc_one:
                    fc1_all.append(k)
                for g in label:
                    y_label.append(g)

                accumulated_mse += mse_value * actual_bsize
                logger.info('evaluation in progress: running mse = %.2f, processed = %d / %d',
                            mse_value, (step - 1) * hyper.batch_size + actual_bsize, total_size)
            duration = default_timer() - start_time
            fc1_all = np.array(fc1_all)
            y_label = np.array(y_label)
            logger.info('======================= TBCNN End =================================')
            logger.info('')
    return fc1_all, y_label


def ml_model(model, name, feature, label):
    logger.info('')
    logger.info('======================= Machine Learning ====================================')
    score_pre = cross_val_score(model, feature, label, scoring='neg_mean_squared_error', cv=10)
    rmse = []
    rmse_all = .0
    for i in score_pre:
        rmse.append(abs(i) ** 0.5)
    for j in rmse:
        rmse_all = rmse_all + j
    logger.info('%s: rmse = %.6f', name, rmse_all / 10)
    return rmse_all/10


def ml_model_mae(model, name, feature, label):
    score_pre = cross_val_score(model, feature, label, scoring='neg_mean_absolute_error', cv=10).mean()
    return score_pre


def concept_get(concept_dir):
    # load the file and read the data
    data = pd.read_csv(concept_dir, encoding='gbk')
    x_csv = data[['single table', 'equivalent and non equivalent connection', 'self join', 'outer join',
                  'multi table connection', 'nested queries with in', 'nested queries with comparator',
                  'nested queries with any', 'nested queries with some', 'nested queries with all',
                  'nested queries with exists', 'union', 'intersect', 'except', 'derived table',
                  'order by', 'group by', 'aggregate function', 'distinct', 'character matching',
                  'number of expressions', 'number of tables', 'number of fields']]
    x_csv = x_csv.values
    return x_csv


def text_get(text_dir):
    # load the file and read the data
    x_data, y_data = [], []
    with open(text_dir, 'r') as f_train:
        for eachline in f_train:
            line = json.loads(eachline)
            x_data.append(list(map(float, line['feature'])))
            y_data.append(float(line['diff']))
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data


def main():
    apputil.initialize(variable_scope='tbcnn')

    # get text feature
    text_file = '../data/otherFeature/word2vec_feature.json'
    text_feature = text_get(text_file)  # numpy.ndarray

    # get code feature
    code_feature, label = tbcnn_svm()  # 'numpy.ndarray

    # concatenate all feature
    feature = np.hstack((text_feature, code_feature))  # 'numpy.ndarray'

    lr_scores, svm_scores, bp_scores, gbdt_scores, rf_scores = [], [], [], [], []
    lr_scores_mae, svm_scores_mae, bp_scores_mae, gbdt_scores_mae, rf_scores_mae = [], [], [], [], []
    for i in range(5):
        lr_name = 'LR'
        lr_model = LinearRegression()
        lr_scores.append(ml_model(lr_model, lr_name, feature, label))
        lr_scores_mae.append(ml_model_mae(lr_model, lr_name, feature, label))

        svm_name = 'SVM'
        svm_model = SVR(kernel='sigmoid', degree=3, C=0.9, max_iter=100000)
        svm_scores.append(ml_model(svm_model, svm_name, feature, label))
        svm_scores_mae.append(ml_model_mae(svm_model, svm_name, feature, label))

        bp_name = 'BPNN'
        bp_model = MLPRegressor()
        bp_scores.append(ml_model(bp_model, bp_name, feature, label))
        bp_scores_mae.append(ml_model_mae(bp_model, bp_name, feature, label))

        gbdt_name = 'GBDT'
        gbdt_model = GradientBoostingRegressor()
        gbdt_scores.append(ml_model(gbdt_model, gbdt_name, feature, label))
        gbdt_scores_mae.append(ml_model_mae(gbdt_model, gbdt_name, feature, label))

        rf_name = 'RF'
        rf_model = RandomForestRegressor(n_estimators=50)
        rf_scores.append(ml_model(rf_model, rf_name, feature, label))
        rf_scores_mae.append(ml_model_mae(rf_model, rf_name, feature, label))

        logger.info('======================= end %d ====================================', i)

    logger.info('LR rmse=%.6f', mean_fun(lr_scores))
    logger.info('LR mae=%.6f', mean_fun(lr_scores_mae))

    logger.info('SVM rmse=%.6f', mean_fun(svm_scores))
    logger.info('SVM mae=%.6f', mean_fun(svm_scores_mae))

    logger.info('BPNN rmse=%.6f', mean_fun(bp_scores))
    logger.info('BPNN mae=%.6f', mean_fun(bp_scores_mae))

    logger.info('GBDT rmse=%.6f', mean_fun(gbdt_scores))
    logger.info('GBDT mae=%.6f', mean_fun(gbdt_scores_mae))

    logger.info('RF rmse=%.6f', mean_fun(rf_scores))
    logger.info('RF mae=%.6f', mean_fun(rf_scores_mae))


def mean_fun(list):
    sum = .0
    for j in range(len(list)):
        sum = sum + list[j]
    return sum/len(list)


if __name__ == '__main__':
    main()

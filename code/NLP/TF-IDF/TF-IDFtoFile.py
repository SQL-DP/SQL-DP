# -*- coding: utf-8 -*-
__author__ = 'SQL-DP'

import json
from collections import defaultdict
import math
import operator
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def feature_select(list_words):
    # Statistics of total word frequency
    doc_frequency = defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i] += 1

    # Calculate the TF value of each word
    word_tf = {}
    for i in doc_frequency:
        word_tf[i] = doc_frequency[i] / sum(doc_frequency.values())

    # Calculate the IDF value of each word
    doc_num = len(list_words)
    word_idf = {}  # 存储每个词的idf值
    word_doc = defaultdict(int)  # 存储包含该词的文档数
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i] += 1
    for i in doc_frequency:
        word_idf[i] = math.log(doc_num / (word_doc[i] + 1))

    # Calculate the TF-IDF value of each word
    word_tf_idf = {}
    for i in doc_frequency:
        word_tf_idf[i] = word_tf[i] * word_idf[i]

    # Sort dictionaries by value from large to small
    dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    return dict_feature_select


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def cut_word(word_original):
    stopwords_path = 'data/stopwords.dat'
    word_cut1 = word_original.strip()
    word_cut2 = word_cut1.replace(' ', '')
    word = word_cut2.replace('\n', '').replace('\r', '')
    cw = jieba.cut(word)
    stopwords = stopwordslist(stopwords_path)
    outstr = ''
    for word in cw:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr


if __name__ == '__main__':
    # load data
    data = pd.read_csv(r'sample_data/question.csv', encoding='gbk')

    # word segmentation and remove stop words
    train_data = []
    ques_text = data['question_detail'].apply(cut_word)
    for i in range(len(ques_text)):
        train_data.append(ques_text[i])
    ques_text_nda = np.array(ques_text)
    difficulty = data['difficulty']

    features = feature_select(ques_text)

    vectorizer = CountVectorizer(max_features=10)
    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(train_data))
    x_train_weight = tf_idf.toarray()

    word_dict = {}
    for i in range(len(ques_text)):
        ques_text_to = [ques_text_nda[i]]
        tf_idf = tf_idf_transformer.transform(vectorizer.transform(ques_text_to))
        x_test_weight = tf_idf.toarray()

        weight_ques_text = x_test_weight.flatten()  # ndarray
        weight = weight_ques_text.tolist()

        word_dict = dict(id=i + 1, feature=weight, diff=difficulty[i])
        # save
        with open('sample_data/TF_IDF_feature.json', "a") as output_file:
            json.dump(word_dict, output_file, ensure_ascii=False)
            output_file.write('\n')
        output_file.close()

    print("Finish!")





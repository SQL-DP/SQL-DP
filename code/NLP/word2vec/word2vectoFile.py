__author__ = 'SQL-DP'

import json
from gensim.models import Word2Vec
import pandas as pd


def is_chinese(uchar):
    if (uchar >= u'\u4e00') and (uchar <= u'\u9fa5'):
        return True
    else:
        return False


if __name__ == '__main__':
    wordNum_exist = 17  # This parameter needs to be calculated according to different datasets
    dim = 50

    # load the trained word2vec model
    model = Word2Vec.load("model/word_embedding_50")

    # load data
    f = open("data/SQLcorpus.txt", encoding="utf-8")
    SQL_question_total = f.readlines()
    f.close()
    data = pd.read_csv(r'question.csv', encoding='gbk')
    difficulty = data['difficulty']

    for i in range(len(SQL_question_total)):
        question_word = SQL_question_total[i].split()
        SQL_vector = []
        for j in question_word:
            if is_chinese(j) is True:
                try:
                    vector = model.wv[j]
                    word_vector = vector.tolist()
                    SQL_vector = SQL_vector + word_vector
                except Exception as e:
                    print("Characters not found: "+j)

        if len(SQL_vector) < (wordNum_exist * dim):
            # When the length of the question vector is less than the fixed length, fill it with 0
            for k in range(wordNum_exist * dim-len(SQL_vector)):
                SQL_vector.append(0)

        if len(SQL_vector) > (wordNum_exist * dim):
            # When the length of the topic vector is greater than the fixed length, it will be deleted
            SQL_vector = SQL_vector.count(wordNum_exist * dim)

        word_dict = dict(id=i + 1, feature=SQL_vector, diff=difficulty[i])

        # Save
        with open('data/word2vec_feature.json', "a") as output_file:
            json.dump(word_dict, output_file, ensure_ascii=False)
            output_file.write('\n')
        output_file.close()

    print("Finish!")
__author__ = 'SQL-DP'

import jieba
import json
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd


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


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


if __name__ == "__main__":
    # load data
    data = pd.read_csv(r'sample_data/question.csv', encoding='gbk')
    ques_text = data['question_detail'].apply(cut_word)
    ques_text_nda = np.array(ques_text)
    difficulty = data['difficulty']

    count = CountVectorizer()
    bag = count.fit_transform(ques_text_nda)
    word_dict = {}
    for i in range(len(ques_text)):
        word_dict = dict(id=i+1, feature=bag.toarray()[i].tolist(), diff=difficulty[i])
        with open('sample_data/BOW_feature.json', "a") as output_file:
            json.dump(word_dict, output_file, ensure_ascii=False)
            output_file.write('\n')
    output_file.close()

    print("Finish!")


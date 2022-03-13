__author__ = 'SQL-DP'

import jieba
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
    data = pd.read_csv(r'question.csv', encoding='gbk')

    # word segmentation and remove stop words
    ques_text = data['question_detail'].apply(cut_word)
    ques_text_nda = np.array(ques_text)

    # save
    f = open('data/SQLcorpus.txt', mode='w', encoding='utf-8')
    for i in range(len(ques_text_nda)):
        f.write(ques_text_nda[i]+'\n')
    f.close()

    print("Finish!")



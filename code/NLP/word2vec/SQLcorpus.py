__author__ = 'SQL-DP'

import jieba
import numpy as np
import pandas as pd


def cut_word(word_original):
    word_cut1 = word_original.strip()
    word_cut2 = word_cut1.replace(' ', '')
    word = word_cut2.replace('\n', '').replace('\r', '')
    cw = jieba.cut(word)
    stopwords = stopwordslist('data/stopwords.dat')
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
    data = pd.read_csv(r'question_text.csv', encoding='gbk')
    x_data = data[['question_detail']]
    x_data = x_data.astype(str)

    # word segmentation and remove stop words
    demo = x_data['question_detail'].apply(cut_word)
    demo_nda = np.array(demo)

    # save
    f = open('data/SQLcorpus.txt', mode='w', encoding='utf-8')
    for i in range(len(demo_nda)):
        f.write(demo_nda[i]+'\n')
    f.close()  # close file

    print("Finish!")



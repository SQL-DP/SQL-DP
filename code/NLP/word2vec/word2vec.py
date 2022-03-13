__author__ = 'SQL-DP'

from gensim.models import Word2Vec, word2vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    # train model
    sentences = LineSentence("data/SQLyuliaoku.txt")
    model = Word2Vec(sentences, vector_size=50, window=5, min_count=5, workers=4, epochs=10)
    
    model.save('model/word_embedding_50')
    
    # for C-MIDP, R-MIDP, and H-MIDP model
    # model.wv.save_word2vec_format("data/word2vec_50.txt", binary=False)

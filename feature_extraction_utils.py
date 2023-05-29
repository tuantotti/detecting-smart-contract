import numpy as np
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# TF-IDF
class TfIdf:

    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        self.vectorized = TfidfVectorizer()

    def __call__(self, *args, **kwargs):
        # create vocabulary and calculate idf
        X_train = self.vectorized.fit_transform(raw_documents=self.X_train).toarray()

        X_test = self.vectorized.transform(raw_documents=self.X_test).toarray()

        return X_train, X_test


class Word2Vec:
    def __init__(self, word_index):
        self.word_index = word_index
        self.fasttext_model = FastText.load('./word2vec/fasttext_model.model')

    def __call__(self, *args, **kwargs):
        vocab_size = len(self.word_index) + 1
        output_dim = 128
        print(vocab_size)
        embedding_matrix = np.random.random((vocab_size, output_dim))
        for word, i in self.word_index.items():
            try:
                embedding_vector = self.fasttext_model.wv[word]
            except:
                print(word, 'not found')
            if embedding_vector is not None:
                embedding_matrix[i, :] = embedding_vector

        return embedding_matrix


class BagOfWord:
    def __init__(self, X_train, X_test):
        self.ngram_range = (1, 1)
        self.X_train, self.X_test = X_train, X_test

        self.vectorizer = CountVectorizer(analyzer='word', input='content', ngram_range=self.ngram_range,
                                          max_features=None)

    def __call__(self, *args, **kwargs):
        X_train_bow = self.vectorizer.fit_transform(self.X_train).toarray()
        X_test_bow = self.vectorizer.transform(self.X_test).toarray()

        return X_train_bow, X_test_bow

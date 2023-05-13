from keras.layers import TextVectorization
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


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
    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, *args, **kwargs):
        weights_df = pd.read_csv('./word2vec/vectors.csv')
        vocab_df = pd.read_csv('./word2vec/vocab.csv').to_numpy().ravel()
        vocab_size = vocab_df.shape[0]

        vectorizer = TextVectorization(max_tokens=vocab_size,
                                       output_mode='int',
                                       output_sequence_length=self.max_length,
                                       vocabulary=vocab_df)

        return vocab_size, vectorizer, weights_df.to_numpy()

from sklearn.feature_extraction.text import TfidfVectorizer


# TF-IDF
class TfIdf:

    def __init__(self, X):
        self.corpus = X
        self.vectorized = TfidfVectorizer()

    def __call__(self, *args, **kwargs):
        # create vocabulary and calculate idf
        self.vectorized.fit(self.corpus)

        self.result = self.vectorized.transform(self.corpus)

        return self.result.toarray()

from sklearn.feature_extraction.text import TfidfVectorizer


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

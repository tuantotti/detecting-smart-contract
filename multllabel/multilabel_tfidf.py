import pandas as pd
import os
import numpy as np

from sklearn.metrics import classification_report, accuracy_score, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from skmultilearn.adapt import MLkNN
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

"""
Read and preprocess data
"""
data_folder = os.getcwd() + '/data-multilabel/'
data = pd.read_csv(data_folder + '/.././report/Data_Cleansing.csv')
data = data.drop(['Unnamed: 0', 'index', 'ADDRESS', 'LABEL_FORMAT'], axis=1)
remove_label = data['LABEL'].value_counts().keys().tolist()[6:]
remove_index = data[data['LABEL'].isin(remove_label)].index
data.drop(remove_index, inplace=True)
X, y = data['BYTECODE'], data.iloc[:, 2:].to_numpy()

def save_classification(y_test,y_pred, out_dir):
  print(classification_report(y_test,y_pred))
  out = classification_report(y_test,y_pred, output_dict=True)
  out_df = pd.DataFrame(out).transpose()
  out_df.to_csv(out_dir)

  return out_df

"""## Feature Extraction
### TF IDF
"""
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

"""## Split data"""
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
tfidf = TfIdf(X_train, X_test)
X_train, X_test = tfidf()

"""# Multilabel machine learning
### Label Powerset
"""
classifier = LabelPowerset(LogisticRegression())
classifier.fit(X_train, Y_train)
y_pred_lbp = classifier.predict(X_test)

save_classification(y_test=Y_test, y_pred=y_pred_lbp, out_dir='.././report/Label_Powerset_TFIDF.csv')

"""### Binary relevence"""
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())
classifier.fit(X_train, Y_train)
y_pred_binre = classifier.predict(X_test)
save_classification(y_test=Y_test, y_pred=y_pred_binre, out_dir='.././report/Binary_Relevence_TFIDF.csv')

"""### Classifier Chains"""
base_lr = LogisticRegression()

chains = [ClassifierChain(base_lr, order="random", random_state=i) for i in range(9)]
for chain in chains:
    chain.fit(X_train, Y_train)

Y_pred_chains = np.array([chain.predict(X_test) for chain in chains])
chain_jaccard_scores = [
    jaccard_score(Y_test, Y_pred_chain >= 0.5, average="samples")
    for Y_pred_chain in Y_pred_chains
]

Y_pred_ensemble = Y_pred_chains.mean(axis=0)
ensemble_jaccard_score = jaccard_score(Y_test, Y_pred_ensemble >= 0.5, average="samples")
save_classification(y_test=Y_test, y_pred=Y_pred_ensemble, out_dir='.././report/Classifier_Chains_TFIDF.csv')

"""### Adapted Algorithm"""
classifier = MLkNN(k=8)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
save_classification(y_test=Y_test, y_pred=Y_pred_ensemble, out_dir='.././report/Adapted_Algorithm_TFIDF.csv')

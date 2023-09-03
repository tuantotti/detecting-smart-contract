import pandas as pd
import os
import numpy as np
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from sklearn.model_selection import train_test_split
from base_multilabel import MultilabelModel
from utils.feature_extraction_utils import TfIdf
from save_report import save_classification

"""
Read and preprocess data
"""
print("Read and preprocess data")
data_folder = os.getcwd() + '/data-multilabel/'
data = pd.read_csv(data_folder + '/Data_Cleansing.csv')
data = data.drop(['Unnamed: 0', 'index', 'ADDRESS', 'LABEL_FORMAT'], axis=1)
num_classes = 4
selected_columns = ['BYTECODE', 'Timestamp dependence', 'Outdated Solidity version', 'Frozen Ether', 'Delegatecall Injection']
data = data.loc[:, selected_columns]
X, y = data['BYTECODE'], data.iloc[:, -num_classes:].to_numpy()
print('y: ', y.shape)
labels = data.iloc[:, -num_classes:].keys().tolist()
values = np.sum(y, axis=0)

print(dict(zip(labels, values)))

"""## Split data"""
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""## Feature Extraction
### TF IDF
"""
print("Feature Extraction - TFIDF")
tfidf = TfIdf(X_train, X_test)
X_train, X_test = tfidf()

"""# Multilabel machine learning
### Label Powerset
"""
print("Multilabel machine learning")
print("Label Powerset")
lbl_powerset = MultilabelModel(X_train=X_train, y_train=Y_train, X_test=X_test, 
                               method='LabelPowerset', num_classes=num_classes)
y_pred_lbp = lbl_powerset()

save_classification(y_test=Y_test, y_pred=y_pred_lbp.toarray(), out_dir='./report/Label_Powerset_TFIDF.csv', labels=labels)

"""### Binary relevence"""
print("Binary relevence")
# with a gaussian naive bayes base classifier
bin_relevence = MultilabelModel(X_train=X_train, y_train=Y_train, X_test=X_test, 
                               method='BinaryRelevance', num_classes=num_classes)
y_pred_binre = bin_relevence()
save_classification(y_test=Y_test, y_pred=y_pred_binre.toarray(), out_dir='./report/Binary_Relevence_TFIDF.csv', labels=labels)

"""### Classifier Chains"""
print("Classifier Chains")
classifier_chain = MultilabelModel(X_train=X_train, y_train=Y_train, X_test=X_test, 
                               method='ClassifierChain', num_classes=num_classes)
Y_pred_chains = classifier_chain()
save_classification(y_test=Y_test, y_pred=Y_pred_chains.astype(int), out_dir='./report/Classifier_Chains_TFIDF.csv', labels=labels)

"""### Adapted Algorithm"""
print("Adapted Algorithm")
adapt_al = MultilabelModel(X_train=X_train, y_train=Y_train, X_test=X_test, 
                               method='MLkNN', num_classes=num_classes)
y_pred_adapt = adapt_al()
save_classification(y_test=Y_test, y_pred=y_pred_adapt.toarray().astype(int), out_dir='./report/Adapted_Algorithm_TFIDF.csv', labels=labels)

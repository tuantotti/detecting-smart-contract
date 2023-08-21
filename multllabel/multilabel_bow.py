import pandas as pd
import os
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from base_multilabel import MultilabelModel
from ..feature_extraction_utils import BagOfWord

"""
Read and preprocess data
"""
print("Read and preprocess data")
data_folder = os.getcwd() + '/data-multilabel/'
data = pd.read_csv(data_folder + '/Data_Cleansing.csv')
data = data.drop(['Unnamed: 0', 'index', 'ADDRESS', 'LABEL_FORMAT'], axis=1)
# remove_label = data['LABEL'].value_counts().keys().tolist()[6:]
# remove_index = data[data['LABEL'].isin(remove_label)].index
# data.drop(remove_index, inplace=True)
num_classes = 4
selected_columns = ['BYTECODE', 'Timestamp dependence', 'Outdated Solidity version', 'Frozen Ether', 'Delegatecall Injection']
data = data.loc[:, selected_columns]
X, y = data['BYTECODE'], data.iloc[:, -num_classes:].to_numpy()
print('y: ', y.shape)
labels = data.iloc[:, -num_classes:].keys().tolist()
values = np.sum(y, axis=0)

print(dict(zip(labels, values)))

def save_classification(y_test,y_pred, out_dir):
  print(classification_report(y_test,y_pred, target_names=labels))
  out = classification_report(y_test,y_pred, output_dict=True, target_names=labels)
  out_df = pd.DataFrame(out).transpose()
  out_df.to_csv(out_dir)

  return out_df

"""## Split data"""
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""## Feature Extraction
### BOW
"""
print("Feature Extraction - Bag Of Word")
bow = BagOfWord(X_train=X_train, X_test=X_test)
X_train_bow, X_test_bow = bow()

"""# Multilabel machine learning
### Label Powerset
"""
print("Multilabel machine learning")
print("Label Powerset")
lbl_powerset = MultilabelModel(X_train=X_train, y_train=Y_train, X_test=X_test, 
                               method='LabelPowerset', num_classes=num_classes)
y_pred_lbp = lbl_powerset()

save_classification(y_test=Y_test, y_pred=y_pred_lbp, out_dir='.././report/Label_Powerset_BOW.csv')

"""### Binary relevence"""
print("Binary relevence")
# with a gaussian naive bayes base classifier
bin_relevence = MultilabelModel(X_train=X_train, y_train=Y_train, X_test=X_test, 
                               method='BinaryRelevance', num_classes=num_classes)
y_pred_binre = bin_relevence()
save_classification(y_test=Y_test, y_pred=y_pred_binre, out_dir='.././report/Binary_Relevence_BOW.csv')

"""### Classifier Chains"""
print("Classifier Chains")
classifier_chain = MultilabelModel(X_train=X_train, y_train=Y_train, X_test=X_test, 
                               method='ClassifierChain', num_classes=num_classes)
Y_pred_chains = classifier_chain()
Y_pred_ensemble = Y_pred_chains.mean(axis=0)
save_classification(y_test=Y_test, y_pred=Y_pred_ensemble.astype(int), out_dir='.././report/Classifier_Chains_BOW.csv')

"""### Adapted Algorithm"""
print("Adapted Algorithm")
adapt_al = MultilabelModel(X_train=X_train, y_train=Y_train, X_test=X_test, 
                               method='MLkNN', num_classes=num_classes)
y_pred_adapt = adapt_al()
save_classification(y_test=Y_test, y_pred=y_pred_adapt.astype(int), out_dir='.././report/Adapted_Algorithm_BOW.csv')

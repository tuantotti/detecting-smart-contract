import collections
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC

import model


class MachineLearningModel(model.Model):
    def __init__(self, X_train, X_test, y_train, y_test, num_class, no_vul_label, num_opcode, input_length, algorithm,
                 weights=None, is_set_weight=True, save_path="./report/bow_lstm_weight.csv",
                 checkpoint_multi_filepath='./best_model_lstm_mi/best_model_multi.hdf5',
                 checkpoint_binary_filepath='./best_model_lstm_mi/best_model_binary.hdf5'):
        super().__init__(X_train, X_test, y_train, y_test, num_class, no_vul_label, num_opcode, input_length,
                         is_set_weight, save_path, checkpoint_multi_filepath, checkpoint_binary_filepath)
        self.algorithm = algorithm
        self.svc = SVC(probability=True, kernel='linear')

    def __call__(self, *args, **kwargs):
        self.run()

    def build_binary_model(self) -> Any:
        if self.algorithm == 'naive_bayes':
            gau = GaussianNB()
            return gau
        elif self.algorithm == 'random_forest':
            rf = RandomForestClassifier()
            return rf
        else:
            ada = AdaBoostClassifier(n_estimators=10, learning_rate=1)
            return ada

    def build_multi_model(self) -> Any:
        if self.algorithm == 'naive_bayes':
            mnb = MultinomialNB()
            return mnb
        elif self.algorithm == 'random_forest':
            rf = RandomForestClassifier()
            return rf
        else:
            ada = AdaBoostClassifier(n_estimators=10, learning_rate=1)
            return ada

    def run(self):
        y_binary_train, y_binary_test, X_vul_train, y_vul_train = self.prepare_data()

        # binary model
        binary_model = self.build_binary_model()
        binary_model.fit(self.X_train, y_binary_train)
        y_pred_binary = binary_model.predict(self.X_test)
        print('classification_report: \n', classification_report(y_binary_test, y_pred_binary))
        print('Confusion Matrix: \n', confusion_matrix(y_binary_test, y_pred_binary))
        y_pred_binary[y_pred_binary == 0.] = self.no_vul_label
        print('y_binary_pred', collections.Counter(y_pred_binary))

        # multi label model
        label_dict = collections.Counter(y_vul_train)
        print('y_vul_train', collections.Counter(y_vul_train))

        vul_index_test = np.where(y_pred_binary == 1.)
        X_vul_test = self.X_test[vul_index_test]
        y_vul_test = self.y_test[vul_index_test]
        print('y_vul_test\n', collections.Counter(y_vul_test))

        multilabel_model = self.build_multi_model()
        multilabel_model.fit(X_vul_train, y_vul_train)
        y_predict = multilabel_model.predict(X_vul_test)
        print('y_predict', collections.Counter(y_predict))
        print('classification_report: \n', classification_report(y_vul_test, y_predict))
        print('Confusion Matrix: \n', confusion_matrix(y_vul_test, y_predict))

        y_result = y_pred_binary.copy()
        j = 0
        for i in range(len(y_result)):
            if y_result[i] != self.no_vul_label:
                y_result[i] = y_predict[j]
                j += 1

        print('classification_report: \n', classification_report(self.y_test, y_result))
        print('Confusion Matrix: \n', confusion_matrix(self.y_test, y_result))
        self.save_result(self.y_test, y_result)

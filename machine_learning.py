import collections
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB, MultinomialNB

import model


class MachineLearningModel(model.Model):
    def __init__(self, X_train, X_test, y_train, y_test, num_class, no_vul_label, num_opcode, input_length, algorithm):
        super().__init__(X_train, X_test, y_train, y_test, num_class, no_vul_label, num_opcode, input_length)
        self.algorithm = algorithm

    def __call__(self, *args, **kwargs):
        self.run()

    def build_binary_model(self) -> Any:
        if self.algorithm == 'naive_bayes':
            gau = GaussianNB()
            return gau
        else:
            rf = RandomForestClassifier()
            return rf

    def build_multi_model(self) -> Any:
        if self.algorithm == 'naive_bayes':
            mnb = MultinomialNB()
            return mnb
        else:
            rf = RandomForestClassifier()
            return rf

    def run(self):
        y_binary_train, y_binary_test, X_vul_train, y_vul_train, X_vul_test = self.prepare_data()

        # binary model
        gau = self.build_binary_model()
        gau.fit(self.X_train, y_binary_train)
        y_binary_pred = gau.predict(self.X_test)
        print('classification_report: \n', classification_report(y_binary_test, y_binary_pred))
        print('Confusion Matrix: \n', confusion_matrix(y_binary_test, y_binary_pred))
        y_binary_pred[y_binary_pred == 0.] = self.no_vul_label
        print('y_binary_pred', collections.Counter(y_binary_pred))

        # multi label model
        # label_dict = collections.Counter(y_vul_train)
        # print('y_vul_train', collections.Counter(y_vul_train))
        # class_weight = self.create_class_weight(label_dict, 0.8)
        # weight = y_vul_train.copy()
        # for i in range(0, 4):
        #     weight[weight == i] = class_weight[i]

        mnb = self.build_multi_model()
        mnb.fit(X_vul_train, y_vul_train)
        y_predict = mnb.predict(X_vul_test)
        print('y_predict', collections.Counter(y_predict))

        y_result = y_binary_pred.copy()
        j = 0
        for i in range(len(y_result)):
            if y_result[i] != self.no_vul_label:
                y_result[i] = y_predict[j]
                j += 1

        print('classification_report: \n', classification_report(y_binary_test, y_result))
        print('Confusion Matrix: \n', confusion_matrix(y_binary_test, y_result))
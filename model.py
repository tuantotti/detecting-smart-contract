import collections
import math
from abc import abstractmethod
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


class Model:
    def __init__(self, X_train, X_test, y_train, y_test, num_class, no_vul_label, num_opcode, input_length):
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.num_class = num_class
        self.no_vul_label = no_vul_label
        self.num_opcode = num_opcode
        self.input_length = input_length
        self.checkpoint_multi_filepath = './best_model_lstm_mi/best_model_multi.hdf5'
        self.checkpoint_binary_filepath = './best_model_lstm_mi/best_model_binary.hdf5'

    def __call__(self, *args, **kwargs):
        self.run()

    @abstractmethod
    def build_binary_model(self) -> Any:
        pass

    @abstractmethod
    def build_multi_model(self) -> Any:
        pass

    @staticmethod
    def create_class_weight(labels_dict, mu):
        """Create weight based on the number of domain name in the dataset"""
        total = np.sum(list(labels_dict.values()))
        keys = labels_dict.keys()
        class_weight = dict()

        for key in keys:
            score = math.pow(total / float(labels_dict[key]), mu)
            class_weight[key] = score

        return class_weight

    def save_confusion_matrix(self, cf_matrix):
        ax = sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,
                         fmt='.2%', cmap='Blues')

        ax.set_title('Confusion Matrix\n\n')
        ax.set_xlabel('\nPredicted Contract Label')
        ax.set_ylabel('Actual Contract Label');
        ax.xaxis.set_ticklabels(['0', '1', '2', '3', '9'])
        ax.yaxis.set_ticklabels(['0', '1', '2', '3', '9'])

        plt.savefig(type(self).__name__ + "_confusion_matrix.png")

    def save_result(self, test, pred):
        out = classification_report(test, pred, output_dict=True)
        out_df = pd.DataFrame(out).transpose()
        out_dir = './report/' + type(self).__name__ + "_" + str(datetime.now()) + '.csv'
        out_df.to_csv(out_dir)

    def prepare_data(self):
        no_vul_binary_label = 0.
        vul_binary_label = 1.

        y_binary_train = self.y_train.copy()
        y_binary_train = np.where(y_binary_train == self.no_vul_label, no_vul_binary_label, vul_binary_label)
        print('y_binary_train\n', collections.Counter(y_binary_train))

        y_binary_test = self.y_test.copy()
        y_binary_test = np.where(y_binary_test == self.no_vul_label, no_vul_binary_label, vul_binary_label)
        print('y_binary_test\n', collections.Counter(y_binary_test))

        # Data for vulnerable classification
        vul_index = np.where(self.y_train != self.no_vul_label)
        X_vul_train = self.X_train[vul_index]
        y_vul_train = self.y_train[vul_index]
        print('y_vul_train\n', collections.Counter(y_vul_train))

        return y_binary_train, y_binary_test, X_vul_train, y_vul_train

    def run(self, max_epoch=10, batch_size=256):
        y_binary_train, y_binary_test, X_vul_train, y_vul_train = self.prepare_data()

        """ Build the model for binary classification """
        model_binary = self.build_binary_model()

        # callback to save model
        binary_callback = ModelCheckpoint(
            filepath=self.checkpoint_binary_filepath,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        # Train model
        model_binary.fit(self.X_train, y_binary_train, batch_size=batch_size, epochs=max_epoch,
                         callbacks=[binary_callback], validation_split=0.1)

        print("========Binary Predict ================")
        best_model_binary = load_model(self.checkpoint_binary_filepath)
        y_pred_binary = best_model_binary.predict(self.X_test)
        y_pred_binary = y_pred_binary.ravel()
        y_pred_binary[y_pred_binary >= 0.5] = 1.
        y_pred_binary[y_pred_binary < 0.5] = self.no_vul_label
        print('y_pred_binary\n', collections.Counter(y_pred_binary))

        print(classification_report(y_binary_test, y_pred_binary))
        print(confusion_matrix(y_binary_test, y_pred_binary))

        if len(collections.Counter(y_pred_binary)) == 1:
            print(classification_report(self.y_test, y_pred_binary))
            print(confusion_matrix(self.y_test, y_pred_binary))

            self.save_confusion_matrix(confusion_matrix(self.y_test, y_pred_binary))

            self.save_result(self.y_test, y_pred_binary)
            return

        """ Build model for multiclass classification """
        labels_dict = collections.Counter(y_vul_train)
        # Create class weight
        class_weight = self.create_class_weight(labels_dict=labels_dict, mu=0.6)
        # Build model
        model_multi = self.build_multi_model()
        # Callback to save the best model
        multilabel_callback = ModelCheckpoint(
            filepath=self.checkpoint_multi_filepath,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        # Train model
        model_multi.fit(X_vul_train, y_vul_train, batch_size=batch_size, epochs=max_epoch,
                        class_weight=class_weight,
                        callbacks=[multilabel_callback], validation_split=0.1)

        """ Load the best model """
        best_model_multi = load_model(self.checkpoint_multi_filepath)

        """ Evaluation """
        vul_index_test = np.where(y_pred_binary == 1.)
        X_vul_test = self.X_test[vul_index_test]
        y_vul_test = self.y_test[vul_index_test]
        print('y_vul_test\n', collections.Counter(y_vul_test))

        y_pred_multi = best_model_multi.predict(X_vul_test)
        y_pred_multi = np.argmax(y_pred_multi, axis=1)
        print('y_pred_multi\n', collections.Counter(y_pred_multi))
        print(classification_report(y_vul_test, y_pred_multi))
        print(confusion_matrix(y_vul_test, y_pred_multi))

        """ Combine by replacing with compatible label """
        y_result = y_pred_binary.copy()
        j = 0
        for i in range(len(y_result)):
            if y_result[i] == 1.:
                y_result[i] = y_pred_multi[j]
                j = j + 1

        print('y_result\n', collections.Counter(y_result))
        """ Result """
        print(classification_report(self.y_test, y_result))
        print(confusion_matrix(self.y_test, y_result))

        self.save_confusion_matrix(confusion_matrix(self.y_test, y_result))

        self.save_result(self.y_test, y_result)

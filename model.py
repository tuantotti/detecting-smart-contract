import collections
import math
from abc import abstractmethod
from typing import Any

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit


class Model:
    def __init__(self, _X, y, num_class, no_vul_label, num_opcode, input_length):
        self.X = _X
        self.y = y
        self.num_class = num_class
        self.no_vul_label = no_vul_label
        self.num_opcode = num_opcode
        self.input_length = input_length

    def __call__(self, *args, **kwargs):
        self.run(max_epoch=1, n_folds=10, batch_size=128)

    @abstractmethod
    def build_binary_model(self) -> Any:
        pass

    @abstractmethod
    def build_multi_model(self) -> Any:
        pass

    def create_class_weight(self, labels_dict, mu):
        """Create weight based on the number of domain name in the dataset"""
        total = np.sum(list(labels_dict.values()))
        keys = labels_dict.keys()
        class_weight = dict()

        for key in keys:
            score = math.pow(total / float(labels_dict[key]), mu)
            class_weight[key] = score

        return class_weight

    def run(self, max_epoch=1, n_folds=1, batch_size=128):
        # binary label
        y_binary = self.y.copy()
        y_binary = np.where(y_binary == self.no_vul_label, 0, 1)
        collections.Counter(y_binary)
        # print(self.y[1082])

        sss = StratifiedShuffleSplit(n_folds, test_size=0.2, random_state=0)
        fold = 0
        for train, test in sss.split(self.X, y_binary, self.y):
            fold = fold + 1
            print(train)
            print("=============FOLD %d================" % fold)
            # X_train, y_train use for evaluating binary model, X_test, y_test for testing binary model
            X_train, X_test, y_train, y_test, y_mul_train, y_mul_test = \
                self.X[train], self.X[test], y_binary[train], y_binary[test], self.y[train], self.y[test]
            # X_vul, y_vul use for multiclass classification
            y_vul = []
            X_vul = []
            # create the multilabel data train by removing the no-vul data
            for i in range(len(y_mul_train)):
                if y_mul_train[i] != self.no_vul_label:
                    X_vul.append(X_train[i])
                    y_vul.append(y_mul_train[i])

            X_vul = np.array(X_vul)
            y_vul = np.array(y_vul)
            print(collections.Counter(y_vul))

            # Build the model for binary classification
            model_binary = self.build_model_binary()

            # callback to save model
            checkpoint_binary_filepath = './best_model_lstm_mi/best_model_binary.hdf5'
            model_checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_binary_filepath,
                monitor='val_loss',
                mode='min',
                save_best_only=True)

            print(collections.Counter(y_train))
            # Train model
            model_binary.fit(X_train, y_train, batch_size=batch_size, epochs=max_epoch,
                             callbacks=[model_checkpoint_callback], validation_split=0.05)
            # Multiclass classification
            # Split data
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=0)
            for train_multi, test_multi in sss2.split(X_vul, y_vul):
                X_train_multi, X_holdout_multi, y_train_multi, y_holdout_multi = X_vul[train_multi], X_vul[test_multi], \
                    y_vul[train_multi], y_vul[test_multi]
            labels_dict = collections.Counter(y_train_multi)
            # Create class weight
            class_weight = self.create_class_weight(labels_dict=labels_dict, mu=0.6)
            # Build model
            model_multi = self.build_multiclass_model()
            # Callback to save the best model
            checkpoint_multi_filepath = './best_model_lstm_mi/best_model_multi.hdf5'
            model_checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_multi_filepath,
                monitor='val_loss',
                mode='min',
                save_best_only=True)
            # Train model
            print(X_train_multi.shape)
            print(collections.Counter(y_train_multi))
            model_multi.fit(X_train_multi, y_train_multi, batch_size=batch_size, epochs=max_epoch,
                            class_weight=class_weight,
                            callbacks=[model_checkpoint_callback], validation_data=(X_holdout_multi, y_holdout_multi))

            # load the best model
            best_model_binary = load_model(checkpoint_binary_filepath)
            best_model_multi = load_model(checkpoint_multi_filepath)

            # calculate confusion matrix and combine to multilabel
            print("Predict ================")
            y_pred_binary = best_model_binary.predict(X_test)
            y_pred_binary = y_pred_binary.ravel()
            y_pred_binary = [1 if (y_pred_binary[x] > 0.5) else self.no_vul_label for x in range(len(y_pred_binary))]
            y_pred_binary = np.array(y_pred_binary)
            print(collections.Counter(y_train))

            X_dga_test = []

            for i in range(len(y_pred_binary)):
                if y_pred_binary[i] == 1:
                    X_dga_test.append(X_test[i])

            X_dga_test = np.array(X_dga_test)
            print(X_dga_test.shape)

            y_pred_multi = best_model_multi.predict(X_dga_test)
            y_pred_multi = np.argmax(y_pred_multi, axis=1)

            # combine by replacing with compatible label
            j = 0
            for i in range(len(y_pred_binary)):
                if y_pred_binary[i] == 1:
                    y_pred_binary[i] = y_pred_multi[j]
                    j = j + 1

            print(y_pred_binary)
            print(y_mul_test)
            print(classification_report(y_pred_binary, y_mul_test))
            # save report to csv
            # classification_report_csv(y_pred_binary, y_mul_test)

            print("=============END FOLD %d=============" % fold)

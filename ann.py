import keras.models
from keras.layers import Dense, Input, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1

from model import Model


class ANN(Model):
    def __init__(self, X_train, X_test, y_train, y_test, num_class, no_vul_label, num_opcode, input_length,
                 weights=None, is_set_weight=True, save_path="./report/bow_lstm_weight.csv",
                 checkpoint_multi_filepath='./best_model_lstm_mi/best_model_multi.hdf5',
                 checkpoint_binary_filepath='./best_model_lstm_mi/best_model_binary.hdf5'):
        super().__init__(X_train, X_test, y_train, y_test, num_class, no_vul_label, num_opcode, input_length,
                         is_set_weight, save_path, checkpoint_multi_filepath, checkpoint_binary_filepath)
        self.weights = weights

    def __call__(self, *args, **kwargs):
        self.run()

    def build_base_model(self, num_output, activation, loss, metric):
        inputs = Input(shape=(self.input_length,))
        dense = Dense(512, activation='relu', kernel_regularizer=l1(0.000001))
        x = dense(inputs)
        x = Dense(256, activation='relu', kernel_regularizer=l1(0.000001))(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu', kernel_regularizer=l1(0.000001))(x)
        x = Dropout(0.2)(x)
        outputs = Dense(num_output, activation=activation)(x)

        _model = keras.models.Model(inputs=inputs, outputs=outputs)
        _model.summary()

        adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        _model.compile(
            optimizer=adam, loss=loss, metrics=[metric]
        )

        return _model

    def build_binary_model(self):
        model = self.build_base_model(num_output=1, activation='sigmoid',
                                      loss='binary_crossentropy', metric='binary_accuracy')

        return model

    def build_multi_model(self):
        model = self.build_base_model(num_output=self.num_class - 1, activation='softmax',
                                      loss='sparse_categorical_crossentropy', metric='accuracy')

        return model

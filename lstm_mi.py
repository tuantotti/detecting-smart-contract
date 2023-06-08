import keras
from keras import Input
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l1

from model import Model


class LstmMiModel(Model):
    def __init__(self, X_train, X_test, y_train, y_test, num_class, no_vul_label, num_opcode, input_length,
                 weights=None, is_set_weight=True, save_path="./report/bow_lstm_weight.csv",
                 checkpoint_multi_filepath='./best_model_lstm_mi/best_model_multi.hdf5',
                 checkpoint_binary_filepath='./best_model_lstm_mi/best_model_binary.hdf5'):
        super().__init__(X_train, X_test, y_train, y_test, num_class, no_vul_label, num_opcode, input_length,
                         is_set_weight, save_path, checkpoint_multi_filepath, checkpoint_binary_filepath)
        self.weights = weights

    def __call__(self, *args, **kwargs):
        self.run()

    def build_base_lstm_model(self, num_output, activation, loss, metric):
        inputs = Input(shape=(self.input_length,))
        if self.weights is None:
            embeddings = Embedding(input_dim=self.num_opcode, output_dim=128)(inputs)
        else:
            embeddings = Embedding(input_dim=self.num_opcode, output_dim=128, weights=[self.weights])(inputs)
        x = LSTM(units=128)(embeddings)
        # x = LSTM(units=128, kernel_regularizer=l1(0.000001), return_sequences=True)(embeddings)
        x = Dropout(rate=0.2)(x)
        # x = LSTM(units=64, kernel_regularizer=l1(0.000001))(x)
        # x = Dropout(rate=0.2)(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(num_output, activation=activation)(x)

        _model = keras.models.Model(inputs=inputs, outputs=outputs)
        _model.summary()

        adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        _model.compile(
            optimizer=adam, loss=loss, metrics=[metric]
        )

        return _model

    def build_binary_model(self):
        model = self.build_base_lstm_model(num_output=1, activation='sigmoid',
                                           loss='binary_crossentropy', metric='binary_accuracy')

        return model

    def build_multi_model(self):
        model = self.build_base_lstm_model(num_output=self.num_class - 1, activation='softmax',
                                           loss='sparse_categorical_crossentropy', metric='accuracy')

        return model

from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential

from model import Model


class LstmMiModel(Model):
    def __init__(self, X_train, X_test, y_train, y_test, num_class, no_vul_label, num_opcode, input_length,
                 vectorizer=None, weights=None):
        super().__init__(X_train, X_test, y_train, y_test, num_class, no_vul_label, num_opcode, input_length)
        self.vectorizer = vectorizer
        self.weights = weights

    def __call__(self, *args, **kwargs):
        self.run()

    def build_base_model(self):
        model = Sequential()
        if self.vectorizer is not None:
            model.add(self.vectorizer)
            model.add(Embedding(input_dim=self.num_opcode, output_dim=128, input_length=self.input_length,
                                weights=self.weights))
        else:
            model.add(Embedding(input_dim=self.num_opcode, output_dim=128, input_length=self.input_length))

        return model

    def build_binary_model(self):
        model = self.build_base_model()

        model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.5, return_sequences=True))
        model.add(LSTM(128, activation='relu', dropout=0.1, recurrent_dropout=0.5))
        model.add(Dense(units=1, activation='sigmoid'))
        model.summary()

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
        return model

    def build_multi_model(self):
        model = self.build_base_model()

        model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.5, return_sequences=True))
        model.add(LSTM(128, activation='relu', dropout=0.1, recurrent_dropout=0.5))
        model.add(Dense(self.num_class - 1, activation='softmax'))

        model.summary()

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

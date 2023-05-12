from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential

from model import Model


class LstmMiModel(Model):
    def __call__(self, *args, **kwargs):
        self.run()

    def build_binary_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.num_opcode, output_dim=128, input_length=self.input_length))
        model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.5, return_sequences=True))
        model.add(LSTM(128, activation='relu', dropout=0.1, recurrent_dropout=0.5))
        model.add(Dense(units=1, activation='sigmoid'))
        model.summary()

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
        return model

    def build_multi_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.num_opcode, output_dim=128, input_length=self.input_length))
        model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.5, return_sequences=True))
        model.add(LSTM(128, activation='relu', dropout=0.1, recurrent_dropout=0.5))
        model.add(Dense(self.num_class - 1, activation='softmax'))

        model.summary()

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

import keras.models
from keras.layers import Dense, Input, Dropout

from model import Model


class ANN(Model):

    def __call__(self, *args, **kwargs):
        self.run()

    def build_base_model(self, num_output, activation, loss, metric):
        inputs = Input(shape=(self.input_length,))
        dense = Dense(128, activation='relu')
        x = dense(inputs)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_output, activation=activation)(x)

        _model = keras.models.Model(inputs=inputs, outputs=outputs)
        _model.summary()

        _model.compile(
            optimizer="adam", loss=loss, metrics=[metric]
        )

        return _model

    def build_binary_model(self):
        model = self.build_base_model(num_output=1, activation='sigmoid',
                                      loss='binary_crossentropy', metric='binary_accuracy')

        return model

    def build_multi_model(self):
        model = self.build_base_model(num_output=self.num_class, activation='softmax',
                                      loss='sparse_categorical_crossentropy', metric='accuracy')

        return model

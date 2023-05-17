import tensorflow as tf
from keras.layers import Embedding, Dense, Layer, MultiHeadAttention, LayerNormalization, Dropout, \
    GlobalAveragePooling1D, Input
from keras.models import Model
from tensorflow import keras

import model


class Transformer(model.Model):
    def __init__(self, X_train, X_test, y_train, y_test, num_class, no_vul_label, num_opcode, input_length, embed_dim,
                 num_heads, ff_dim):
        super().__init__(X_train, X_test, y_train, y_test, num_class, no_vul_label, num_opcode, input_length)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.checkpoint_multi_filepath = './transformer_model/best_model_multi'
        self.checkpoint_binary_filepath = './transformer_model/best_model_binary'

    def base_model(self, num_output, activation, loss):
        inputs = Input(shape=(self.input_length,))

        # Embedding layer
        embedding_layer = TokenAndPositionEmbedding(self.input_length, self.num_opcode, self.embed_dim)
        x = embedding_layer(inputs)
        # Transformer layer
        transformer_block = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
        x = transformer_block(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.3)(x)

        # Feed-forward layer
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_output, activation=activation)(x)

        _model = Model(inputs=inputs, outputs=outputs)
        _model.summary()

        _model.compile(
            optimizer="adam", loss=loss, metrics=["accuracy"]
        )

        return _model

    def build_binary_model(self):
        return self.base_model(num_output=1, activation='sigmoid', loss="binary_crossentropy")

    def build_multi_model(self):
        return self.base_model(num_output=self.num_class, activation='softmax',
                               loss="sparse_categorical_crossentropy")


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def __call__(self, x):
        maxlength = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlength, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)

        return x + positions


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim), ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

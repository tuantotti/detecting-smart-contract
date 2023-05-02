import tensorflow as tf
from keras.layers import Embedding, Dense, Layer, MultiHeadAttention, LayerNormalization, Dropout, \
    GlobalAveragePooling1D, Input
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tensorflow import keras

import model


class Transformer(model.Model):
    def __init__(self, _X, y, num_class, no_vul_label, num_opcode, input_length, embed_dim, num_heads, ff_dim):
        super().__init__(_X, y, num_class, no_vul_label, num_opcode, input_length)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

    def base_model(self, num_output, activation):
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
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        return _model

    def build_binary_model(self):
        return self.base_model(num_output=1, activation='sigmoid')

    def build_multi_model(self):
        return self.base_model(num_output=self.num_class - 1, activation='softmax')


class PreprocessData:
    def __init__(self, dataset, _vocab_size, _max_sequence_length):
        self.dataset = dataset
        self.vocab_size = _vocab_size
        self.max_sequence_length = _max_sequence_length

    def nlp_preprocess(self, data):
        tokenizer = Tokenizer(num_words=self.vocab_size, lower=False)

        tokenizer.fit_on_texts(data['BYTECODE'].values)

        sequences = tokenizer.texts_to_sequences(data['BYTECODE'].values)

        _X = pad_sequences(sequences, maxlen=self.max_sequence_length)

        return _X

    def format_value(self, data, value):
        data['LABEL'] = value
        data = data[data['BYTECODE'].apply(lambda x: str(type(x)) == "<class 'str'>")]
        return data


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

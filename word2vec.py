import tensorflow as tf
from keras.layers import Embedding
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.preprocessing.sequence import make_sampling_table, skipgrams
from keras.preprocessing.text import Tokenizer

from utils_method import read_data


class Word2Vec(tf.keras.Model):
    def __init__(self, _vocab_size, _embedding_dim, _num_ns):
        super(Word2Vec, self).__init__()
        self.target_embedding = Embedding(_vocab_size,
                                          _embedding_dim,
                                          input_length=1,
                                          name="w2v_embedding")
        self.context_embedding = Embedding(_vocab_size,
                                           _embedding_dim,
                                           input_length=_num_ns + 1)

    def __call__(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        # dots: (batch, context)
        return dots


# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(_sequences, window_size, _num_ns, _vocab_size, seed):
    # Elements of each training example are appended to these lists.
    _targets, _contexts, _labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = make_sampling_table(_vocab_size)

    # Iterate over all sequences (sentences) in the dataset.
    for sequence in _sequences:

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = skipgrams(
            sequence,
            vocabulary_size=_vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0
        )

        # Iterate over each positive skip-gram pair to produce training examples
        # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=_num_ns,
                unique=True,
                range_max=_vocab_size,
                seed=seed,
                name="negative_sampling"
            )

            # Build context and label vectors (for one target word)
            context = tf.concat([tf.squeeze(context_class, 1), negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * _num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            _targets.append(target_word)
            _contexts.append(context)
            _labels.append(label)

    return _targets, _contexts, _labels


dataset = read_data()
X, y = dataset['BYTECODE'], dataset['LABEL']

vocab_size = 300
BATCH_SIZE = 1024
BUFFER_SIZE = 10000
num_ns = 4
embedding_dim = 64

tokenizer = Tokenizer(num_words=vocab_size, lower=False)
# Create vocabulary
tokenizer.fit_on_texts(X.values)
# Transforms each text in texts to a sequence of integers
sequences = tokenizer.texts_to_sequences(X.values)

targets, contexts, labels = generate_training_data(
    _sequences=sequences,
    window_size=2,
    _num_ns=num_ns,
    _vocab_size=vocab_size,
    seed=42)

dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer=Adam(),
                 loss=CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
word2vec.fit(dataset, epochs=20)

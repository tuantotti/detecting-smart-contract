import os

import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

from transformer import Transformer


def nlp_preprocess(X, max_length, num_opcode):
    tokenizer = Tokenizer(num_words=num_opcode, lower=False)

    # Create vocabulary
    tokenizer.fit_on_texts(X.values)

    # Transforms each text in texts to a sequence of integers
    sequences = tokenizer.texts_to_sequences(X.values)

    # Pads sequences to the same length
    _X = pad_sequences(sequences, maxlen=max_length)

    return _X


def read_data():
    directory = os.getcwd() + '/data/'
    label_dict = {}
    listdir = os.listdir(directory)
    dataset = pd.DataFrame(columns=['BYTECODE', 'LABEL'])
    i = 0
    for _dir in listdir:
        p = pd.read_csv(directory + _dir, usecols=['BYTECODE'])
        if p.shape[0] > 1000:
            if _dir == 'Contracts_No_Vul.csv':
                p['LABEL'] = len(listdir) - 1
            else:
                p['LABEL'] = i
                i += 1

            label_dict[p.loc[0, 'LABEL']] = _dir
            p = p[p['BYTECODE'].apply(lambda x: str(type(x)) == "<class 'str'>")]
            dataset = pd.concat([dataset, p])
    print(label_dict)
    return dataset


def run():
    # Preprocessing
    vocab_size = 300
    max_length = 5500

    print("================Read data================")
    dataset = read_data()
    directory = os.getcwd() + '/data/'
    listdir = os.listdir(directory)
    no_vul_label = len(listdir) - 1

    X, y = dataset['BYTECODE'], dataset['LABEL']
    print(y.shape)
    print(y.value_counts())
    y = y.to_numpy()
    num_class = len(y.value_counts())

    print(num_class)
    print(no_vul_label)

    # Feature extraction
    print("================Feature extraction================")
    X = nlp_preprocess(X, max_length, num_opcode=vocab_size)
    # TF-IDF
    # tf = TfIdf(X)
    # X_tfidf = tf()
    # print(X_tfidf.shape)
    # vocab_size = X_tfidf.shape[1]
    #
    # print(X_tfidf.shape)

    # pad to fix-length input

    # Classification
    print("================Classification================")
    embed_dim, num_heads, ff_dim = 64, 1, 128
    trans = Transformer(X, y, num_class, no_vul_label, vocab_size, max_length, embed_dim, num_heads, ff_dim)
    trans()
    # Use lstm model
    # lstm_mi = LstmMiModel(X_tfidf, y, num_class=num_class, no_vul_label=no_vul_label, num_opcode=vocab_size,
    #                       input_length=X_tfidf.shape[1])
    # lstm_mi()


if __name__ == '__main__':
    run()

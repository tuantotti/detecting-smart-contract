import os

import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


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


def nlp_preprocess(X, max_length, num_opcode):
    tokenizer = Tokenizer(num_words=num_opcode, lower=False)

    # Create vocabulary
    tokenizer.fit_on_texts(X.values)

    # Transforms each text in texts to a sequence of integers
    sequences = tokenizer.texts_to_sequences(X.values)

    # Pads sequences to the same length
    _X = pad_sequences(sequences, maxlen=max_length)

    return _X

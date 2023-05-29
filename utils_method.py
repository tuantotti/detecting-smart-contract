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
        if _dir == 'Contracts_No_Vul.csv':
            p.drop_duplicates(inplace=True)
            p['LABEL'] = float(len(listdir) - 1)
        else:
            # p = p[:9704]
            p['LABEL'] = float(i)
            i += 1

        label_dict[p.loc[0, 'LABEL']] = _dir.split('.csv')[0]
        p = p[p['BYTECODE'].apply(lambda x: str(type(x)) == "<class 'str'>")]
        dataset = pd.concat([dataset, p])
    
    label_dict = dict(sorted(label_dict.items()))
    
    return dataset, label_dict


def nlp_preprocess(X, max_length):
    tokenizer = Tokenizer(lower=False)

    # Create vocabulary
    tokenizer.fit_on_texts(X)

    # Transforms each text in texts to a sequence of integers
    sequences = tokenizer.texts_to_sequences(X)

    # Pads sequences to the same length
    _X = pad_sequences(sequences, maxlen=max_length)

    return _X, tokenizer.word_index

import os
from sklearn.metrics import classification_report, accuracy_score

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


def save_classification(y_test,y_pred, out_dir, labels):
  out = classification_report(y_test,y_pred, output_dict=True, target_names=labels)
  total_support = out['samples avg']['support']
  accuracy = accuracy_score(y_test, y_pred)
  out['accuracy'] = {'precision': accuracy, 'recall': accuracy, 'f1-score': accuracy, 'support': total_support}
  out_df = pd.DataFrame(out).transpose()
  print(out_df)

  out_df.to_csv(out_dir)

  return out_df

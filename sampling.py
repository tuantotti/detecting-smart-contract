import time
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

bytecode_vul_data = np.array([
    # {
    #     'label': 'Authentication through tx.origin.csv',
    #     'value': 1,
    # },
    {
        'label': 'Delegatecall Injection.csv',
        'value': 0,
    },
    {
        'label': 'Frozen Ether.csv',
        'value': 1,
    },
    {
        'label': 'Leaking Ether to arbitrary address.csv',
        'value': 2,
    },
    {
        'label': 'Timestamp dependence.csv',
        'value': 3,
    },
    # {
    #     'label': 'Ugradeable contract.csv',
    #     'value': 5,
    # },
    # {
    #     'label': 'Unprotected Suicide.csv',
    #     'value': 7,
    # },
    # {
    #     'label': 'Outdated Solidity version.csv',
    #     'value': 8,
    # },
])
NUM_CLASS = len(bytecode_vul_data) + 1
NO_VUL_LABEL = len(bytecode_vul_data)
label_dict = np.array([
    {
        'label': 'No vulnerability',
        'value': '4',
    },
    {
        'label': 'Delegate call Injection',
        'value': '0',
    },
    {
        'label': 'Frozen Ether',
        'value': '1',
    },
    {
        'label': 'Leaking Ether to arbitrary address',
        'value': '2',
    },
    {
        'label': 'Timestamp dependence',
        'value': '3',
    },
    # {
    #     'label': 'Upgradeable contract',
    #     'value': '5',
    # }
])

CONTRACT_TYPE_COLUMN = "Contract Type"
NUMBER_OF_RECORD_COLUMN = "Number of records"


def nlp_preprocess(df):
    n_most_common_opcodes = 1000  # 8000
    max_len = 130

    tokenizer = Tokenizer(num_words=n_most_common_opcodes, lower=False)

    tokenizer.fit_on_texts(df['BYTECODE'].values)

    sequences = tokenizer.texts_to_sequences(df['BYTECODE'].values)

    # word_index = tokenizer.word_index
    # print('Found %s unique tokens.' % len(word_index))

    _X = pad_sequences(sequences, maxlen=max_len)

    return _X


def df_to_xy(df, numClass):
    _X = nlp_preprocess(df)
    _y = to_categorical(df['LABEL'], num_classes=numClass)

    return _X, _y


def format_value(df, value):
    df['LABEL'] = value
    df = df[df['BYTECODE'].apply(lambda x: str(type(x)) == "<class 'str'>")]
    return df


def process_data():
    no_vul_contract = 'Contracts_No_Vul.csv'
    _non_vul_data = pd.read_csv('./data/' + no_vul_contract, usecols=['OPCODE', 'CATEGORY'])
    _non_vul_data.columns = ['BYTECODE', 'LABEL']
    _non_vul_data['LABEL'] = NO_VUL_LABEL

    formatData = []
    for data in bytecode_vul_data:
        p = pd.read_csv('./data/' + data['label'], usecols=['BYTECODE', 'LABEL'])
        if int(data['value']) == 0:
            p = p[:35171]
        if int(data['value']) == 1:
            p = p[:97359]
        if int(data['value']) == 2:
            p = p[:595]
        if int(data['value']) == 3:
            p = p[:54666]
        p = format_value(p, data['value'])
        formatData.append(p)

    _vul_data = pd.concat(formatData)

    # concatenate vulnerable and non-vulnerable into one set
    _dataset = pd.concat([_vul_data, _non_vul_data], ignore_index=True)
    print(_dataset['LABEL'].value_counts())
    return _dataset


def train_predict_model(_X_train, _X_test, _y_train, _y_test, epochs=20, emb_dim=128, batch_size=256):
    print((_X_train.shape, _y_train.shape, _X_test.shape, _y_test.shape))
    n_most_common_opcodes = 1000

    model = Sequential()
    model.add(Embedding(n_most_common_opcodes, emb_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.1))
    model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(len(bytecode_vul_data) + 1, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())

    start_time = time.time()

    # training the model
    model.fit(_X_train, _y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    end_time = time.time()
    print('Time taken: ', end_time - start_time)

    predict_x = model.predict(_X_test)
    _y_pred = np.argmax(predict_x, axis=1)

    _y_test_format = np.argmax(_y_test, axis=1)

    return _y_test_format, _y_pred


def over_sampling(_X_train, _y_train):
    over = SMOTE(random_state=0)

    _x_over_sampling, _y_over_sampling = over.fit_resample(_X_train, _y_train)
    return _x_over_sampling, _y_over_sampling


def under_sampling(_X_train, _y_train):
    nm = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = nm.fit_resample(X, y)
    return X_resampled, y_resampled


def hybrid_sampling(_X_train, _y_train):
    smote_enn = SMOTETomek(random_state=42)
    _X_combine, _y_combine = smote_enn.fit_resample(X, y)
    return _X_combine, _y_combine


def classification_report_csv(_y_test_format, _y_pred, report_name, _label_dict=label_dict):
    report = classification_report(_y_test_format, _y_pred, output_dict=True)
    print('\n classification report:\n', classification_report(_y_test_format, _y_pred))
    df_classification_report = pd.DataFrame(report).transpose()

    for _i in _label_dict:
        df_classification_report.rename(index={_i['value']: _i['label']}, inplace=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    df_classification_report.to_csv('./report/' + report_name + timestamp + '.csv', index=True)


def save_to_csv(_label_dict=label_dict):
    analysis_data = dataset['LABEL'].value_counts()
    analysis_data = pd.DataFrame({
        CONTRACT_TYPE_COLUMN: analysis_data.index,
        NUMBER_OF_RECORD_COLUMN: analysis_data.values
    })
    for i in _label_dict:
        analysis_data.loc[analysis_data[CONTRACT_TYPE_COLUMN] == int(i['value']), CONTRACT_TYPE_COLUMN] = i['label']
    analysis_data.to_csv('./report/analysis_data_' + time.strftime("%Y%m%d-%H%M%S") + '.csv', index=False)


# split processed dataset into training and test
print("=============PROCESS DATA================")
dataset = process_data()  # read data

# analysis data and save to file
save_to_csv()
# preprocessing data
X, y = df_to_xy(dataset, len(bytecode_vul_data) + 1)
# # no sampling
# print("=============NO SAMPLING=================")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# y_test_format, y_pred = train_predict_model(X_train, X_test, y_train, y_test)
# classification_report_csv(y_test_format, y_pred, "no_sampling_", label_dict)
# under-sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("Before sampling: ", Counter(np.argmax(y_train, axis=1)))
X_under_sampling, y_under_sampling = under_sampling(X_train, y_train)
print("After sampling: ", Counter(np.argmax(y_under_sampling, axis=1)))
y_test_format, y_pred = train_predict_model(X_under_sampling, X_test, y_under_sampling, y_test)
classification_report_csv(y_test_format, y_pred, "Random_Under_Sampling_")
# over-sampling
# print("=============OVER SAMPLING================")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# x_over_sampling, y_over_sampling = over_sampling(X_train, y_train)
# y_test_format, y_pred = train_predict_model(x_over_sampling, X_test, y_over_sampling, y_test)
# classification_report_csv(y_test_format, y_pred, "SMOTE_")
# # hybrid sampling
# print("=============HYBRID SAMPLING=============")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# x_hybrid, y_hybrid = hybrid_sampling(X_train, y_train)
# y_test_format, y_pred = train_predict_model(x_hybrid, X_test, y_hybrid, y_test)
# classification_report_csv(y_test_format, y_pred, "hybrid_sampling_SMOTETomek_")


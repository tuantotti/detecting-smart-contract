import os

import pandas as pd

directory = os.getcwd() + '/data/'
listdir = os.listdir(directory)


def init_label_dict():
    bytecode = {}

    i = 1
    for _dir in listdir:
        if _dir == 'Contracts_No_Vul.csv':
            bytecode[_dir] = 0
        else:
            bytecode[_dir] = i
            i += 1
    bytecode = dict(sorted(bytecode.items(), key=lambda x: x[1]))
    return bytecode


def remove_duplicates_file():
    _directory = os.getcwd() + '/data/'
    label_dict = {}
    _listdir = os.listdir(_directory)
    dataset = pd.DataFrame(columns=['ADDRESS', 'BYTECODE', 'LABEL'])
    i = 0
    for _dir in _listdir:
        p = pd.read_csv(_directory + _dir)
        if 'CATEGORY' in p.columns:
            p.rename(columns={"OPCODE": "BYTECODE"}, inplace=True)
        if 'Unnamed: 0' in p.columns:
            p.drop(['Unnamed: 0'], axis='columns', inplace=True)

        if 'CATEGORY' in p.columns:
            p.drop(['CATEGORY'], axis='columns', inplace=True)

        if _dir == 'Contracts_No_Vul.csv':
            p['LABEL'] = len(_listdir) - 1
        else:
            # if _dir == 'Outdated Solidity version.csv':
            #     p = p[:500000]
            p['LABEL'] = i
            i += 1

        label_dict[p.loc[0, 'LABEL']] = _dir
        p = p[p['BYTECODE'].apply(lambda x: str(type(x)) == "<class 'str'>")]
        dataset = pd.concat([dataset, p])

    print("Before: ", dataset.shape)
    dataset.drop_duplicates(subset='ADDRESS', inplace=True)
    print(dataset['LABEL'].value_counts())
    print("After: ", dataset.shape)

    for index in label_dict:
        p = dataset.loc[dataset['LABEL'] == index]
        p.drop(['LABEL'], axis='columns', inplace=True)
        p.to_csv(_directory + label_dict[index], index=False)

    print(label_dict)
    return dataset


def init_label():
    dataset = pd.DataFrame(columns=['BYTECODE', 'LABEL'])
    i = 1
    for _dir in listdir:
        p = pd.read_csv(directory + _dir)
        p.drop(['LABEL'], axis='columns', inplace=True)
        if _dir == 'Contracts_No_Vul.csv':
            p['LABEL'] = 0
        else:
            p['LABEL'] = i
            i += 1
        p.to_csv(directory + _dir, index=False)

    return dataset


# dataset = read_data()
remove_duplicates_file()
# remove_duplicates_file()
# p = pd.read_csv(directory + 'Contracts_No_Vul.csv')
# print(p.columns)
# p.drop(['CATEGORY'], axis='columns', inplace=True)
# p['LABEL'] = 0
# p.to_csv(directory + 'Contracts_No_Vul.csv', index=False)

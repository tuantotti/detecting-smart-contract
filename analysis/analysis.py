import matplotlib.pyplot as plt
import os
import pandas as pd

def read_data():
  directory = os.getcwd() + '/data/'
  listdir = os.listdir(directory)
  dataset = pd.DataFrame(columns=['BYTECODE', 'LABEL'])
  for dir in listdir:
    if dir == 'Contracts_No_Vul.csv':
      p = pd.read_csv(directory +dir, usecols=['OPCODE'])
      p.columns = ['BYTECODE']
      p = p[p['BYTECODE'].apply(lambda x: str(type(x)) == "<class 'str'>")]
    else:
      p = pd.read_csv(directory+ dir, usecols=['BYTECODE'])
    p = p[p['BYTECODE'].apply(lambda x: str(type(x)) == "<class 'str'>")]
    dataset = pd.concat([dataset, p])

  return dataset


def analysis_length(dataset):
  number_of_record = len(dataset)
  number_of_record
  title = 'Analysis from '+ str(number_of_record) + ' records'
  x = dataset['BYTECODE'].apply(lambda bytecode : len(bytecode.replace(' ', '')))
  a = x.value_counts()
  a.to_csv('analysis_result.csv',index=True)
  a = a[:100]
  a = a.sort_index()

  fig, ax = plt.subplots()
  ax = a.plot(ax=ax, kind='bar', figsize=(30,15), title=title)
  ax.set_xlabel("Length of bytecode")
  ax.set_ylabel("Frequency")
  plt.savefig('analysis_result.png')

dataset = read_data()
analysis_length(dataset)

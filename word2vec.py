from gensim.models import FastText

from utils_method import read_data

print("Read data")
dataset, label_dict = read_data()
X, y = dataset['BYTECODE'], dataset['LABEL']
sentences = [sentence.split() for sentence in X]
output_dim = 128
model = FastText(vector_size=output_dim, window=6, min_count=1, sentences=sentences, epochs=20)
model.save('./word2vec/fasttext_model.model')

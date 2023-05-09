import os

from transformer import Transformer
from utils_method import read_data, nlp_preprocess


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

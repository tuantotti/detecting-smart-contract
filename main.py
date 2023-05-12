import os

from sklearn.model_selection import train_test_split

from feature_extraction_utils import TfIdf
from lstm_mi import LstmMiModel
from machine_learning import MachineLearningModel
from transformer import Transformer
from utils_method import read_data


def run():
    vocab_size = 300
    max_length = 5500

    print("================Read data================")
    dataset = read_data()
    directory = os.getcwd() + '/data/'
    listdir = os.listdir(directory)
    no_vul_label = float(len(listdir) - 1)

    X, y = dataset['BYTECODE'], dataset['LABEL']
    print(y.shape)
    print(y.value_counts())
    num_class = len(y.value_counts())
    X = X.to_numpy()
    y = y.to_numpy()

    print(num_class)
    print(no_vul_label)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature extraction
    print("================Feature extraction================")
    # X = nlp_preprocess(X, max_length, num_opcode=vocab_size)
    # TF-IDF
    tf = TfIdf(X_train=X_train, X_test=X_test)
    X_train_tfidf, X_test_tfidf = tf()
    vocab_size = X_train_tfidf.shape[1]

    # pad to fix-length input

    # Classification
    print("================Classification================")
    print("================MachineLearningModel naive_bayes================")
    ml = MachineLearningModel(X_train_tfidf, X_test_tfidf, y_train, y_test,
                              num_class=num_class, no_vul_label=no_vul_label,
                              num_opcode=vocab_size, input_length=max_length, algorithm='naive_bayes')
    ml()

    print("================MachineLearningModel random_forest================")
    ml = MachineLearningModel(X_train_tfidf, X_test_tfidf, y_train, y_test,
                              num_class=num_class, no_vul_label=no_vul_label,
                              num_opcode=vocab_size, input_length=max_length, algorithm='random_forest')
    ml()

    print("================LstmMiModel================")
    lstm_mi = LstmMiModel(X_train_tfidf, X_test_tfidf, y_train, y_test,
                          num_class=num_class, no_vul_label=no_vul_label,
                          num_opcode=vocab_size, input_length=max_length)
    lstm_mi()

    print("================Transformer================")
    embed_dim, num_heads, ff_dim = 64, 1, 128
    trans = Transformer(X_train_tfidf, X_test_tfidf, y_train, y_test,
                        num_class=num_class, no_vul_label=no_vul_label,
                        num_opcode=vocab_size, input_length=max_length,
                        embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
    trans()


if __name__ == '__main__':
    run()

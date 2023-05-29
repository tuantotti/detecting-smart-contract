import os

from sklearn.model_selection import train_test_split

from feature_extraction_utils import Word2Vec
from lstm_mi import LstmMiModel
from machine_learning import MachineLearningModel
from utils_method import read_data, nlp_preprocess


def run():
    print("================Read data================")
    dataset, label_dict = read_data()
    directory = os.getcwd() + '/data/'
    listdir = os.listdir(directory)
    no_vul_label = float(len(listdir) - 1)

    X, y = dataset['BYTECODE'], dataset['LABEL']
    print(y.shape)
    print(y.value_counts())
    num_class = len(y.value_counts())
    X = X.to_numpy()
    y = y.to_numpy()

    max_length = 5500
    X_tokenized, word_index = nlp_preprocess(X, max_length)
    vocab_size = len(word_index) + 1

    print(num_class)
    print(no_vul_label)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_tokenized, y, test_size=0.2, random_state=42)

    # Feature extraction
    print("================Feature extraction================")
    # TF-IDF
    word2vec = Word2Vec(word_index)
    embedding_matrix = word2vec()

    print(embedding_matrix.shape)

    # Classification
    print("================Classification================")
    # print("================MachineLearningModel Adaboost================")
    # ml = MachineLearningModel(X_train, X_test, y_train, y_test,
    #                           num_class=num_class, no_vul_label=no_vul_label,
    #                           num_opcode=vocab_size, input_length=max_length, algorithm='adaboost')
    # ml()
    #
    # print("================MachineLearningModel Adaboost================")
    # ml = MachineLearningModel(X_train, X_test, y_train, y_test,
    #                           num_class=num_class, no_vul_label=no_vul_label,
    #                           num_opcode=vocab_size, input_length=max_length, algorithm='adaboost')
    # ml()

    print("================LstmMiModel with weight================")
    lstm_mi = LstmMiModel(X_train, X_test, y_train, y_test,
                          num_class=num_class, no_vul_label=no_vul_label,
                          num_opcode=vocab_size, input_length=max_length, weights=embedding_matrix,
                          save_path="./report/fasttext_lstm_weight.csv")
    lstm_mi()
    print("================LstmMiModel with no weight================")
    lstm_mi = LstmMiModel(X_train, X_test, y_train, y_test,
                          num_class=num_class, no_vul_label=no_vul_label,
                          num_opcode=vocab_size, input_length=max_length, weights=embedding_matrix, is_set_weight=False,
                          save_path="./report/fasttext_lstm_no_weight.csv")
    lstm_mi()


if __name__ == '__main__':
    run()

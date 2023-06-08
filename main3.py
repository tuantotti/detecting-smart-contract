import os

from sklearn.model_selection import train_test_split

from feature_extraction_utils import Word2Vec
from lstm_mi import LstmMiModel
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
    y = y.to_numpy().astype('float')

    max_length = 5500
    X_tokenized, word_index = nlp_preprocess(X, max_length)
    vocab_size = len(word_index) + 1

    print(X_tokenized.shape)

    print(num_class)
    print(no_vul_label)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_tokenized, y, test_size=0.2, random_state=42)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Feature extraction
    print("================Feature extraction================")
    # TF-IDF
    word2vec = Word2Vec(word_index)
    embedding_matrix = word2vec()

    # Classification
    print("================Classification================")
    # print("================MachineLearningModel Naive Bayes================")
    # ml = MachineLearningModel(X_train, X_test, y_train, y_test,
    #                           num_class=num_class, no_vul_label=no_vul_label,
    #                           num_opcode=vocab_size, input_length=max_length, algorithm='naive_bayes',
    #                           save_path="./report/fasttext_nb.csv")
    # ml()
    #
    # rint("================MachineLearningModel Random Forest================")
    # ml = MachineLearningModel(X_train, X_test, y_train, y_test,
    #                           num_class=num_class, no_vul_label=no_vul_label,
    #                           num_opcode=vocab_size, input_length=max_length, algorithm='naive_bayes',
    #                           save_path="./report/fasttext_rf.csv")
    # ml()
    #
    # print("================MachineLearningModel Adaboost================")
    # ml = MachineLearningModel(X_train, X_test, y_train, y_test,
    #                           num_class=num_class, no_vul_label=no_vul_label,
    #                           num_opcode=vocab_size, input_length=max_length, algorithm='adaboost',
    #                           save_path="./report/fasttext_adaboost.csv")
    # ml()

    # print("================LstmMiModel with weight================")
    # lstm_mi = LstmMiModel(X_train, X_test, y_train, y_test,
    #                       num_class=num_class, no_vul_label=no_vul_label,
    #                       num_opcode=vocab_size, input_length=max_length, weights=embedding_matrix,
    #                       save_path="./report/fasttext_lstm_weight.csv",
    #                       checkpoint_multi_filepath='./best_model_lstm_mi/best_model_multi_wor2vec.model',
    #                       checkpoint_binary_filepath='./best_model_lstm_mi/best_model_binary_wor2vec.model')
    # lstm_mi()
    print("================LstmMiModel with no weight================")
    lstm_mi = LstmMiModel(X_train, X_test, y_train, y_test,
                          num_class=num_class, no_vul_label=no_vul_label,
                          num_opcode=vocab_size, input_length=max_length, weights=embedding_matrix, is_set_weight=False,
                          save_path="./report/fasttext_lstm_no_weight.csv",
                          checkpoint_multi_filepath='./best_model_lstm_mi/best_model_multi_wor2vec_no_weight.model',
                          checkpoint_binary_filepath='./best_model_lstm_mi/best_model_binary_wor2vec_no_weight.model')
    lstm_mi()


if __name__ == '__main__':
    run()

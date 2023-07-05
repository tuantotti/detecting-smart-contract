import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ann import ANN
from feature_extraction_utils import Word2Vec
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

    # Feature extraction
    print("================Feature extraction================")
    # TF-IDF
    word2vec = Word2Vec(word_index)
    embedding_matrix = word2vec()

    print(embedding_matrix.shape)

    mean_embedding = embedding_matrix.mean(axis=1)
    print()
    X_mean_embedding = X_tokenized.copy().astype('float32')
    for i, x in enumerate(X_tokenized):
        for j, value in enumerate(x):
            X_mean_embedding[i, j] = mean_embedding[value]

    scaler = MinMaxScaler()
    X_mean_embedding = scaler.fit_transform(X_mean_embedding)
    X_train, X_test, y_train, y_test = train_test_split(X_mean_embedding, y, test_size=0.2, random_state=42)

    # Classification
    print("================Classification================")
    # print("================MachineLearningModel Naive Bayes================")
    # ml = MachineLearningModel(X_train, X_test, y_train, y_test,
    #                           num_class=num_class, no_vul_label=no_vul_label,
    #                           num_opcode=vocab_size, input_length=max_length, algorithm='naive_bayes',
    #                           save_path="./report/fasttext_nb.csv")
    # ml()
    #
    print("================MachineLearningModel Random Forest================")
    ml = MachineLearningModel(X_train, X_test, y_train, y_test,
                              num_class=num_class, no_vul_label=no_vul_label,
                              num_opcode=vocab_size, input_length=max_length, algorithm='random_forest',
                              save_path="./report/fasttext_rf_hyperparameter.csv")
    ml()

    # print("================MachineLearningModel Adaboost================")
    # ml = MachineLearningModel(X_train, X_test, y_train, y_test,
    #                           num_class=num_class, no_vul_label=no_vul_label,
    #                           num_opcode=vocab_size, input_length=max_length, algorithm='adaboost',
    #                           save_path="./report/fasttext_adaboost.csv")
    # ml()

    # print("================ANN Model================")
    # ann = ANN(X_train, X_test, y_train, y_test,
    #           num_class=num_class, no_vul_label=no_vul_label,
    #           num_opcode=vocab_size, input_length=max_length,
    #           save_path="./report/fasttext_ann.csv",
    #           checkpoint_multi_filepath='./best_model_ann/best_model_multi_wor2vec.model',
    #           checkpoint_binary_filepath='./best_model_ann/best_model_binary_wor2vec.model')
    # ann()


if __name__ == '__main__':
    run()

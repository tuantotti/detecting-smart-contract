import os

from sklearn.model_selection import train_test_split

from lstm_mi import LstmMiModel
from utils.feature_extraction_utils import BagOfWord
from utils.utils_method import read_data


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

    print(num_class)
    print(no_vul_label)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature extraction
    print("================Feature extraction================")
    # TF-IDF
    bow = BagOfWord(X_train=X_train, X_test=X_test)
    X_train_bow, X_test_bow = bow()
    vocab_size = max(max(x) for x in X_train_bow) + 1
    max_length = X_train_bow.shape[1]

    # pad to fix-length input

    # Classification
    print("================Classification================")
    print("================LstmMiModel with weight================")
    lstm_mi = LstmMiModel(X_train_bow, X_test_bow, y_train, y_test,
                          num_class=num_class, no_vul_label=no_vul_label,
                          num_opcode=vocab_size, input_length=max_length,
                          save_path="./report/bow_lstm_weight.csv",
                          checkpoint_multi_filepath='./trained/bow/best_model_multi_bow.model',
                          checkpoint_binary_filepath='./trained/bow/best_model_binary_bow.model')
    lstm_mi()

    # print("================LstmMiModel no weight================")
    # lstm_mi = LstmMiModel(X_train_bow, X_test_bow, y_train, y_test,
    #                       num_class=num_class, no_vul_label=no_vul_label,
    #                       num_opcode=vocab_size, input_length=max_length, is_set_weight=False,
    #                       save_path="./report/bow_lstm_no_weight.csv",
    #                       checkpoint_multi_filepath='./best_model_lstm_mi/best_model_multi_bow_no_weight.hdf5',
    #                       checkpoint_binary_filepath='./best_model_lstm_mi/best_model_binary_bow_no_weight.hdf5')
    # lstm_mi()


if __name__ == '__main__':
    run()

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from save_report import save_classification

if torch.cuda.is_available():
 dev = "cuda:0"
else:
 dev = "cpu"
device = torch.device(dev)
device

class Tokenizer(object):
    def __init__(self, num_words=None, lower=True) -> None:
        self.word_index = {}
        self.word_counts = {}
        self.num_words = num_words
        self.split = " "
        self.lower = lower

    def fit_on_texts(self, texts):
        """
        create vocabulary

        Args:
            text: list of strings or list of list of strings
        """
        for text in texts:
            seq = self.text_to_word_sequence(text)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1

                else:
                    self.word_counts[w] = 1
        vocab = self.word_counts.keys()
        self.word_index = dict(zip(vocab, list(range(1, len(vocab) + 1))))

    def text_to_word_sequence(self, input_text):
        if self.lower == True:
            input_text = input_text.lower()

        seq = input_text.split(self.split)
        return seq

    def texts_to_sequences(self, texts):
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts):
        for text in texts:
            seq = self.text_to_word_sequence(text)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                vect.append(i)
            yield vect

def pad_sequences(
    sequences,
    maxlen=None,
    dtype="int32",
    padding="pre",
    truncating="pre",
    value=0.0
):
    """
    Args:
        sequences: List of sequences (each sequence is a list of integers).
        maxlen: Optional Int, maximum length of all sequences. If not provided,
            sequences will be padded to the length of the longest individual
            sequence.
        dtype: (Optional, defaults to `"int32"`). Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, "pre" or "post" (optional, defaults to `"pre"`):
            pad either before or after each sequence.
        truncating: String, "pre" or "post" (optional, defaults to `"pre"`):
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value. (Optional, defaults to 0.)

    Returns:
        Numpy array with shape `(len(sequences), maxlen)`

    Raises:
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """

    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError as e:
            raise ValueError(
                "`sequences` must be a list of iterables. "
                f"Found non-iterable: {str(x)}"
            ) from e

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(
        dtype, np.unicode_
    )
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError(
            f"`dtype` {dtype} is not compatible with `value`'s type: "
            f"{type(value)}\nYou should set `dtype=object` for variable length "
            "strings."
        )

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                f"Shape of sample {trunc.shape[1:]} of sequence at "
                f"position {idx} is different from expected shape "
                f"{sample_shape}"
            )

        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc) :] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')
    return x

"""## Create Model"""
class Branch(nn.Module):
  def __init__(self, INPUT_SIZE, hidden1_size, hidden2_size, dropout, num_outputs):
    super(Branch, self).__init__()

    self.dense1 = nn.Linear(INPUT_SIZE, hidden1_size)
    self.dense2 = nn.Linear(hidden1_size, hidden2_size)
    self.dense3 = nn.Linear(hidden2_size, num_outputs)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
    out_dense1 = self.dense1(x)
    out_dropout = self.dropout(out_dense1)
    out_dense2 = self.dense2(out_dropout)
    out_dense3 = self.dense3(out_dense2)

    return out_dense3

class Escort(nn.Module):
  def __init__(self, vocab_size, embedd_size, gru_hidden_size, n_layers, num_classes):
    super(Escort, self).__init__()
    self.word_embeddings = nn.Embedding(vocab_size, embedd_size)
    self.gru = nn.GRU(embedd_size, gru_hidden_size, num_layers=n_layers)
    self.branches = nn.ModuleList([Branch(gru_hidden_size, 128, 64, 0.2, 1) for _ in range(num_classes)])
    self.sigmoid = nn.Sigmoid()

  def forward(self, sequence):
    embeds = self.word_embeddings(sequence)
    gru_out, _ = self.gru(embeds)
    output_branches = [branch(gru_out[:, -1, :]) for branch in self.branches]
    output_branches = torch.cat(output_branches, dim=1)
    outputs = self.sigmoid(output_branches)
    return outputs

"""## Train model"""
"""### Train and Validation Steps"""

def calculate_score(y_true, preds):
    acc_score = accuracy_score(y_true, preds)

    return acc_score

def train_steps(training_loader, model, loss_f, optimizer):
    training_loss = 0
    nb_tr_steps = 0
    train_acc = 0.

    model.train()
    for step, batch in enumerate(training_loader):
        # push the batch to gpu
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        preds = model(inputs)

        loss = loss_f(preds, labels)
        training_loss += loss.item()

        preds = preds.detach().cpu().numpy()
        preds = np.where(preds>=0.5, 1, 0)
        labels = labels.to('cpu').numpy()

        acc_score = calculate_score(labels, preds)
        train_acc += acc_score

        nb_tr_steps += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # When using GPU
        optimizer.step()

    epoch_loss = training_loss / nb_tr_steps
    epoch_acc = train_acc / nb_tr_steps
    return epoch_loss, epoch_acc

def evaluate_steps(validating_loader, model, loss_f):
    # deactivate dropout layers
    model.eval()

    total_loss = 0

    # empty list to save the model predictions
    total_preds = []
    total_labels = []
    # iterate over batches
    for step, batch in enumerate(validating_loader):
        # push the batch to gpu
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(inputs)

            # compute the validation loss between actual and predicted values
            loss = loss_f(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()
            preds = np.where(preds>=0.5, 1, 0)
            total_preds += list(preds)
            total_labels += labels.tolist()
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(validating_loader)
    acc_score = calculate_score(total_labels, total_preds)

    return avg_loss, acc_score

"""### Training loop"""
def train(EPOCHS, model, optimizer, criterion, dataloader):
  data_train_loader, data_val_loader = dataloader
  # empty lists to store training and validation loss of each epoch
  # set initial loss to infinite
  best_valid_loss = float('inf')
  train_losses = []
  valid_losses = []
  train_accuracies = []
  valid_accuracies = []

  for epoch in range(EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_steps(data_train_loader, model, criterion, optimizer)

    valid_loss, valid_acc = evaluate_steps(data_val_loader, model, criterion)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './trained/escort.pt')
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_acc)
    valid_accuracies.append(valid_acc)

    elapsed_time = time.time() - start_time

    print('Epoch {}/{} \t loss={:.4f} \t accuracy={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'.format(epoch + 1, EPOCHS, train_loss, train_acc, valid_loss, valid_acc, elapsed_time))
  return train_accuracies, valid_accuracies, train_losses, valid_losses

def plot_graph(EPOCHS, train, valid, tittle):
    fig = plt.figure(figsize=(12,12))
    plt.title(tittle)
    plt.plot(list(np.arange(EPOCHS) + 1) , train, label='train')
    plt.plot(list(np.arange(EPOCHS) + 1), valid, label='validation')
    plt.xlabel('num_EPOCHS', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')

"""## Test Model"""

def predict(testing_loader, model):
    # deactivate dropout layers
    model.eval()

    # empty list to save the model predictions
    total_preds = []
    total_labels = []
    start_time = time.time()
    # iterate over batches
    for step, batch in enumerate(testing_loader):
        # push the batch to gpu
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(inputs)

            preds = preds.detach().cpu().numpy()
            preds = np.where(preds>=0.5, 1, 0)
            total_preds += list(preds)
            total_labels += labels.tolist()

    execution_time = (time.time() - start_time) / len(total_labels)
    return total_preds, total_labels, execution_time

def run():
    data_folder = os.getcwd() + '/data-multilabel/'
    data = pd.read_csv(data_folder + 'Data_Cleansing.csv')
    selected_columns = ['BYTECODE', 'Timestamp dependence', 'Outdated Solidity version', 'Frozen Ether', 'Delegatecall Injection']
    data = data.loc[:, selected_columns]
    labels = data.iloc[:, -4:].keys().tolist()
    test = data.iloc[:, -4:].to_numpy()
    values = np.sum(test, axis=0)
    print(dict(zip(labels, values)))
    X, y = data['BYTECODE'], data.iloc[:, -4:].to_numpy()
    BATCH_SIZE = 32
    EMBEDDED_SIZE = 5
    GRU_HIDDEN_SIZE = 64
    NUM_OUTPUT_NODES = 4
    NUM_LAYERS = 1
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    INPUT_SIZE = 4100

    """
    Tokenize data and create vocabulary
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts=X)
    sequences = tokenizer.texts_to_sequences(texts=X)
    X = pad_sequences(sequences, maxlen=INPUT_SIZE)

    """## Split data"""
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=2023)

    tensor_X_train = torch.tensor(X_train)
    tensor_X_val = torch.tensor(X_val)
    tensor_X_test = torch.tensor(X_test)
    tensor_Y_train = torch.FloatTensor(Y_train)
    tensor_Y_val = torch.FloatTensor(Y_val)
    tensor_Y_test = torch.FloatTensor(Y_test)

    train_dataset = TensorDataset(tensor_X_train, tensor_Y_train)
    val_dataset = TensorDataset(tensor_X_val, tensor_Y_val)
    test_dataset = TensorDataset(tensor_X_test, tensor_Y_test)

    data_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    data_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    data_test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    """## Run"""
    SIZE_OF_VOCAB = len(tokenizer.word_index.keys())
    model = Escort(SIZE_OF_VOCAB, EMBEDDED_SIZE, GRU_HIDDEN_SIZE, NUM_LAYERS, NUM_OUTPUT_NODES)
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    train_accuracies, valid_accuracies, train_losses, valid_losses = train(EPOCHS, model, optimizer, criterion, (data_train_loader, data_val_loader))

    plot_graph(EPOCHS, train_losses, valid_losses, "Train/Validation Loss")
    plot_graph(EPOCHS, train_accuracies, valid_accuracies, "Train/Validation Accuracy")

    total_preds, total_labels, execution_time = predict(data_test_loader, model, criterion)

    print('Execution time: ', execution_time)
    save_classification(y_pred=np.array(total_preds), y_test=np.array(total_labels), labels=labels, out_dir='./report/escort.csv')

if __name__ == '__main__':
   run()
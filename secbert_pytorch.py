import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification

from utils_method import read_data

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
print("device: ", device)


class OpcodeData(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.X = X
        self.targets = y
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        text = str(self.X[index])

        inputs = self.tokenizer(
            text,
            None,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }


def prepare_data(dataset):
    X = dataset['BYTECODE']
    y = dataset['LABEL']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    y_binary_train = y_train.copy()
    y_binary_train.loc[y_binary_train != 9] = 1
    y_binary_train.loc[y_binary_train == 9] = 0

    y_binary_val = y_val.copy()
    y_binary_val.loc[y_binary_val != 9] = 1
    y_binary_val.loc[y_binary_val == 9] = 0

    y_binary_test = y_test.copy()
    y_binary_test.loc[y_binary_test != 9] = 1
    y_binary_test.loc[y_binary_test == 9] = 0

    vul_index = y_train.loc[y_train == 9].index
    X_vul_train = X_train.drop(index=vul_index)
    y_vul_train = y_train.drop(index=vul_index)

    vul_index = y_val.loc[y_val == 9].index
    X_vul_val = X_val.drop(index=vul_index)
    y_vul_val = y_val.drop(index=vul_index)

    vul_index = y_test.loc[y_test == 9].index
    X_vul_test = X_test.drop(index=vul_index)
    y_vul_test = y_test.drop(index=vul_index)

    train = (X_train.reset_index(drop=True), y_binary_train.reset_index(drop=True), X_vul_train.reset_index(drop=True),
             y_vul_train.reset_index(drop=True))
    val = (X_val.reset_index(drop=True), y_binary_val.reset_index(drop=True), X_vul_val.reset_index(drop=True),
           y_vul_val.reset_index(drop=True))
    test = (X_test.reset_index(drop=True), y_binary_test.reset_index(drop=True), X_vul_test.reset_index(drop=True),
            y_vul_test.reset_index(drop=True), y_test.reset_index(drop=True))

    return train, val, test


def calculate_score(y_true, preds):
    F1_score = f1_score(y_true, preds, average='macro')
    acc_score = accuracy_score(y_true, preds)

    return acc_score, F1_score


def train_steps(training_loader, model, loss_f, optimizer):
    print('Training...')
    training_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    train_acc = 0.
    train_f1 = 0.

    model.train()

    for step, batch in enumerate(training_loader):
        # push the batch to gpu
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        targets = batch['targets'].to(device)
        # ids, mask, token_type_ids, targets = batch

        outputs = model(ids, attention_mask=mask, token_type_ids=token_type_ids)
        logits = outputs.logits

        loss = loss_f(logits, targets)
        training_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        label_ids = targets.to('cpu').numpy()

        acc_score, F1_score = calculate_score(label_ids, preds)
        train_acc += acc_score
        train_f1 += F1_score
        nb_tr_steps += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # # When using GPU
        optimizer.step()

    epoch_loss = training_loss / nb_tr_steps
    epoch_acc = train_acc / nb_tr_steps
    epoch_f1 = train_f1 / nb_tr_steps
    print(" Accuracy: ", epoch_acc)
    print(" F1 score: ", epoch_f1)
    print(" Average training loss: ", epoch_loss)
    return epoch_loss, epoch_acc, epoch_f1


def evaluate_steps(validating_loader, model, loss_f):
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []
    total_labels = []
    # iterate over batches
    for step, batch in enumerate(validating_loader):
        # push the batch to gpu
        b_input_ids = batch['ids'].to(device)
        b_input_mask = batch['mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        b_labels = batch['targets'].to(device)

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            output = model(b_input_ids, attention_mask=b_input_mask, token_type_ids=token_type_ids)
            logits = output.logits

            # compute the validation loss between actual and predicted values
            loss = loss_f(logits, b_labels)

            total_loss = total_loss + loss.item()

            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
            total_preds += list(preds)
            total_labels += b_labels.tolist()
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(validating_loader)
    acc_score, F1_score = calculate_score(total_labels, total_preds)

    print(" Accuracy: ", acc_score)
    print(" F1 score: ", F1_score)
    print(" Average training loss: ", avg_loss)
    return avg_loss, acc_score, F1_score


def predict(testing_loader, model, loss_f):
    print("\nEvaluating...")
    # deactivate dropout layers
    model.eval()

    # empty list to save the model predictions
    total_preds = []
    total_labels = []
    # iterate over batches
    for step, batch in enumerate(testing_loader):
        # push the batch to gpu
        b_input_ids = batch['ids'].to(device)
        b_input_mask = batch['mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        b_labels = batch['targets'].to(device)

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            output = model(b_input_ids, attention_mask=b_input_mask, token_type_ids=token_type_ids)
            logits = output.logits

            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
            total_preds += list(preds)
            total_labels += b_labels.tolist()

    print(classification_report(total_labels, total_preds))
    return total_preds


def train_binary(binary_secBERT_encoder, training_loader, validating_loader, optimizer):
    print('Train Binary Model')
    binary_secBERT_encoder.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    for epoch in range(EPOCHS):
        print('\nEpoch ', (epoch + 1), '/', EPOCHS)
        train_loss, train_acc, train_f1 = train_steps(training_loader, binary_secBERT_encoder, loss_function,
                                                      optimizer)
        valid_loss, valid_acc, valid_f1 = evaluate_steps(validating_loader, binary_secBERT_encoder, loss_function)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(binary_secBERT_encoder.state_dict(), './trained/secbert/binary_saved_weights.pt')

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        return train_losses, valid_losses


def train_multi(multilabel_secBERT_encoder, training_loader, validating_loader, optimizer):
    print('Train Multilabel Classification')
    multilabel_secBERT_encoder.to(device)
    for param in multilabel_secBERT_encoder.bert.encoder.layer[:3].parameters():
        param.requires_grad_ = False

    loss_function = torch.nn.CrossEntropyLoss()
    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    for epoch in range(EPOCHS):
        print('\nEpoch ', (epoch + 1), '/', EPOCHS)
        train_loss, train_acc, train_f1 = train_steps(training_loader, multilabel_secBERT_encoder, loss_function,
                                                      optimizer)
        valid_loss, valid_acc, valid_f1 = evaluate_steps(validating_loader, multilabel_secBERT_encoder,
                                                         loss_function)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(multilabel_secBERT_encoder.state_dict(), './trained/secbert/multi_saved_weights.pt')

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        return train_losses, valid_losses


# read data
dataset, label_dict = read_data()
print(dataset["LABEL"].value_counts())
# define constant
num_class = len(dataset["LABEL"].value_counts())
max_length = 512
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-05

# load pretrained model
secBERT_tokenizer = BertTokenizerFast.from_pretrained("jackaduma/SecBERT",
                                                      do_lower_case=True)
multilabel_secBERT_encoder = BertForSequenceClassification.from_pretrained("jackaduma/SecBERT",
                                                                           num_labels=num_class - 1)
binary_secBERT_encoder = BertForSequenceClassification.from_pretrained("jackaduma/SecBERT", num_labels=2)

# prepare data, train split data for training, validating, testing
train, val, test = prepare_data(dataset)

X_train, y_binary_train, X_vul_train, y_vul_train = train
X_val, y_binary_val, X_vul_val, y_vul_val = val
X_test, y_binary_test, X_vul_test, y_vul_test, y_test = test

# convert string to index data (create vocabulary, ...)
binary_training_set = OpcodeData(X_train, y_binary_train, secBERT_tokenizer, max_length)
binary_validating_set = OpcodeData(X_val, y_binary_val, secBERT_tokenizer, max_length)
binary_testing_set = OpcodeData(X_test, y_binary_test, secBERT_tokenizer, max_length)

vul_training_set = OpcodeData(X_vul_train, y_vul_train, secBERT_tokenizer, max_length)
vul_validating_set = OpcodeData(X_vul_val, y_vul_val, secBERT_tokenizer, max_length)
vul_testing_set = OpcodeData(X_vul_test, y_vul_test, secBERT_tokenizer, max_length)

# create data generator for train, validate, test
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

val_params = {'batch_size': VALID_BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0
              }

test_params = {'batch_size': VALID_BATCH_SIZE,
               'shuffle': False,
               'num_workers': 0
               }

# Create generator for Dataset with BATCH_SIZE
binary_training_loader = DataLoader(binary_training_set, **train_params)
binary_validating_loader = DataLoader(binary_validating_set, **val_params)
binary_testing_loader = DataLoader(binary_testing_set, **test_params)

# Create generator for Dataset with BATCH_SIZE
vul_training_loader = DataLoader(vul_training_set, **train_params)
vul_validating_loader = DataLoader(vul_validating_set, **val_params)
vul_testing_loader = DataLoader(vul_testing_set, **test_params)

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
binary_optimizer = torch.optim.Adam(params=binary_secBERT_encoder.parameters(), lr=LEARNING_RATE)
vul_optimizer = torch.optim.Adam(params=binary_secBERT_encoder.parameters(), lr=LEARNING_RATE)

# Train data
# train_binary(binary_secBERT_encoder, binary_training_loader, binary_validating_loader, binary_optimizer)
train_multi(multilabel_secBERT_encoder, vul_training_loader, vul_validating_loader, vul_optimizer)

binary_secBERT_encoder.load_state_dict(torch.load('./trained/secbert/binary_saved_weights.pt'))
multilabel_secBERT_encoder.load_state_dict(torch.load('./trained/secbert/multi_saved_weights.pt'))

binary_pred = predict(binary_testing_loader, binary_secBERT_encoder, loss_function)
binary_pred = np.array(binary_pred, dtype=int)
print(classification_report(y_binary_test.to_numpy().astype(int), binary_pred.astype(int)))
df_pred = pd.DataFrame(binary_pred, columns=['LABELS'])
y_vul_test = y_test.loc[df_pred['LABELS'] == 1].reset_index(drop=True)
X_vul_test = X_test.loc[y_vul_test.index].reset_index(drop=True)
vul_testing_set = OpcodeData(X_vul_test, y_vul_test, secBERT_tokenizer, max_length)
vul_testing_loader = DataLoader(vul_testing_set, **test_params)

vul_pred = predict(vul_testing_loader, multilabel_secBERT_encoder, loss_function)

df_pred[df_pred == 0] = 9
result = df_pred.to_numpy()
j = 0
for i in range(len(result)):
    if result[i] == 1:
        result[i] = vul_pred[j]
        j += 1

print(classification_report(y_test.to_numpy().astype(int), result))

out = classification_report(y_test.to_numpy().astype(int), result, output_dict=True)
out_df = pd.DataFrame(out).transpose()
out_df.to_csv('./report/secrobert.csv')

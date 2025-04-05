import argparse
import copy
import sys
import time
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter

torch.use_deterministic_algorithms(True)

OPTS = None

INPUT_DIM = 3 # 
NUM_CLASSES = 10  # Number of classes we are classifying over

def load_data(csv_path, max_rows=None):
    df = pd.read_csv(csv_path, nrows=max_rows)

    # Build card vocab
    all_cards = set()
    for hand in df["hand"]:
        all_cards.update(hand.split(","))
    all_cards.update(df["played_card"])
    all_cards.update(df["top_card"])
    card_vocab = {card: i for i, card in enumerate(sorted(all_cards))}

    NUM_CLASSES = len(card_vocab)
    INPUT_DIM = NUM_CLASSES + 1 + 3

    def encode_row(row):
        x = np.zeros(INPUT_DIM)
        hand_cards = [c for c in row["hand"].split(",") if c != row["played_card"]]
        for card in hand_cards:
            if card in card_vocab:
                x[card_vocab[card]] += 1
        x[len(card_vocab)] = card_vocab.get(row["top_card"], -1)
        x[-3:] = [row["p2"], row["p3"], row["p4"]]
        y = card_vocab.get(row["played_card"], -1)
        return x, y

    X = []
    y = []
    for _, row in df.iterrows():
        x_vec, y_val = encode_row(row)
        X.append(x_vec)
        y.append(y_val)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return X_tensor, y_tensor, INPUT_DIM, NUM_CLASSES


class ThreeLayerMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=200, dropout_prob=0.0):
        super(ThreeLayerMLP, self).__init__()

        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        print(input_dim, hidden_dim)

    def forward(self, x):
        x_mapped = self.input_to_hidden(x)
        x_relu = F.relu(x_mapped)
        x_relu_dropout = self.dropout(x_relu)

        x_mapped = self.hidden_to_hidden(x_relu_dropout)
        x_relu = F.relu(x_mapped)
        x_relu_dropout = self.dropout(x_relu)

        output = self.hidden_to_output(x_relu_dropout)
        output = F.softmax(output, dim=1)

        return output
    
def train(model, X_train, y_train, X_dev, y_dev, lr=1e-1, batch_size=32, num_epochs=30):
    """Run the training loop for the model.

    All of this code is highly generic and works for any model that does multi-class classification.

    Args:
        model: A nn.Module model, must take in inputs of size (B, D)
               and output predictions of size (B, C)
        X_train: Tensor of size (N, D)
        y_train: Tensor of size (N,)
        X_dev: Tensor of size (N_dev, D). Used for early stopping.
        y_dev: Tensor of size (N_dev,). Used for early stopping.
        lr: Learning rate for SGD
        batch_size: Desired batch size.
        num_epochs: Number of epochs of SGD to run
    """
    start_time = time.time()
    loss_func = nn.CrossEntropyLoss()  # (QUESTION 4a: line 1)
                    # Cross-entropy loss is just softmax regression loss
    optimizer = optim.SGD(model.parameters(), lr=lr)  # (QUESTION 4a: line 2)
                    # Stochastic gradient descent optimizer

    # Prepare the training dataset
    # Pytorch DataLoader expects a dataset to be a list of (x, y) pairs
    train_dataset = [(X_train[i,:], y_train[i]) for i in range(len(y_train))] # (QUESTION 4a: line 3)

    # Simple version of early stopping: save the best model checkpoint based on dev accuracy
    best_dev_acc = -1 # (QUESTION 4a: line 4)
    best_checkpoint = None # (QUESTION 4a: line 5)
    best_epoch = -1 # (QUESTION 4a: line 6)

    for t in range(num_epochs): # (QUESTION 4a: line 7)
        train_num_correct = 0 # (QUESTION 4a: line 8)

        # Training loop
        model.train()  # (QUESTION 4a: line 9)
                    # Set model to "training mode", e.g. turns dropout on if you have dropout layers
        for batch in DataLoader(train_dataset, batch_size=batch_size, shuffle=True): # (QUESTION 4a: line 10)
                    # DataLoader automatically groups the data into batchse of roughly batch_size
                    # shuffle=True makes it so that the batches are randomly chosen in each epoch
            x_batch, y_batch = batch  # (QUESTION 4a: line 11)
                    # unpack batch, which is a tuple (x_batch, y_batch)
                    # x_batch is tensor of size (B, D)
                    # y_batch is tensor of size (B,)
            optimizer.zero_grad()  #(QUESTION 4a: line 12)
                    # Reset the gradients to zero
                    # Recall how backpropagation works---gradients are initialized to zero and then accumulated
                    # So we need to reset to zero before running on a new batch!
            logits = model(x_batch) #(QUESTION 4a: line 13)
                    # tensor of size (B, C), each row is the logits (pre-softmax scores) for the C classes
                    # For MNIST, C=10
            loss = loss_func(logits, y_batch)  #(QUESTION 4a: line 14)
                    # Compute the loss of the model output compared to true labels
            loss.backward()  # (QUESTION 4a: line 15)
                    # Run backpropagation to compute gradients
            optimizer.step() # (QUESTION 4a: line 16)
                    # Take a SGD step
                    # Note that when we created the optimizer, we passed in model.parameters()
                    # This is a list of all parameters of all layers of the model
                    # optimizer.step() iterates over this list and does an SGD update to each parameter

            # Compute running count of number of training examples correct
            preds = torch.argmax(logits, dim=1) # (QUESTION 4a: line 17)
                    # Choose argmax for each row (i.e., collapse dimension 1, hence dim=1)
            train_num_correct += torch.sum(preds == y_batch).item() # (QUESTION 4a: line 18)

        # Evaluate train and dev accuracy at the end of each epoch
        train_acc = train_num_correct / len(y_train) # (QUESTION 4a: line 19)
        model.eval()  # (QUESTION 4a: line 20)
                    # Set model to "eval mode", e.g. turns dropout off if you have dropout layers.
        with torch.no_grad():  # (QUESTION 4a: line 21)
                    # Don't allocate memory for storing gradients, more efficient when not training
            dev_logits = model(X_dev) # (QUESTION 4a: line 22)
            dev_preds = torch.argmax(dev_logits, dim=1) # (QUESTION 4a: line 23)
            dev_acc = torch.mean((dev_preds == y_dev).float()).item() # (QUESTION 4a: line 24)
            if dev_acc > best_dev_acc:  # (QUESTION 4a: line 25)
                # Save this checkpoint if it has best dev accuracy so far
                best_dev_acc = dev_acc # (QUESTION 4a: line 26)
                best_checkpoint = copy.deepcopy(model.state_dict()) # (QUESTION 4a: line 27)
                best_epoch = t # (QUESTION 4a: line 28)
        print(f'Epoch {t: <2}: train_acc={train_acc:.5f}, dev_acc={dev_acc:.5f}') # (QUESTION 4a: line 29)

    # Set the model parameters to the best checkpoint across all epochs
    model.load_state_dict(best_checkpoint) # (QUESTION 4a: line 30)
    end_time = time.time()  # (QUESTION 4a: line 31)
    print(f'Training took {end_time - start_time:.2f} seconds') # (QUESTION 4a: line 32)
    print(f'\nBest epoch was {best_epoch}, dev_acc={best_dev_acc:.5f}') # (QUESTION 4a: line 33)

if __name__ == "__main__":
    # Load and preprocess a limited amount of data for testing
    X_tensor, y_tensor, INPUT_DIM, NUM_CLASSES = load_data("../uno_dataset.csv", max_rows=100000)

    # Train/dev/test split
    total = len(X_tensor)
    train_end = int(0.7 * total)
    dev_end = int(0.85 * total)

    X_train, X_dev, X_test = X_tensor[:train_end], X_tensor[train_end:dev_end], X_tensor[dev_end:]
    y_train, y_dev, y_test = y_tensor[:train_end], y_tensor[train_end:dev_end], y_tensor[dev_end:]

    # Initialize model with correct dimensions
    model = ThreeLayerMLP(INPUT_DIM, NUM_CLASSES, hidden_dim=200, dropout_prob=0.5)

    # Train the model
    train(model, X_train, y_train, X_dev, y_dev, lr=0.01, batch_size=32, num_epochs=30)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_preds = torch.argmax(test_logits, dim=1)
        test_acc = torch.mean((test_preds == y_test).float()).item()
    print(f'Test accuracy: {test_acc:.5f}')

    label_counts = Counter(y_tensor.tolist())
    print("Class distribution (label index: count):")
    cnt = 0
    for label, count in sorted(label_counts.items()):
        cnt += count
    for label, count in sorted(label_counts.items()):
        print(f"Class {label}: {count} samples : {count/cnt * 100}%")
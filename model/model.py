import argparse
import copy
import sys
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

torch.use_deterministic_algorithms(True)

OPTS = None

# Todo: Change values for INPUT_DIM & NUM_CLASSES
INPUT_DIM = 784  # = 28 * 28, total size of vector
NUM_CLASSES = 10  # Number of classes we are classifying over

class ThreeLayerMLP(nn.Module):
    def __init__(self, hidden_dim=200, dropout_prob=0.0):
        super(ThreeLayerMLP, self).__init__()

        ### BEGIN_SOLUTION 4e
        self.input_to_hidden = nn.Linear(INPUT_DIM, hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, NUM_CLASSES)
        self.dropout = nn.Dropout(dropout_prob)
        ### END_SOLUTION 4e

    def forward(self, x):
        """Output the predicted scores for each class.

        The outputs are the scores *before* the softmax function.

        Inputs:
            x: Torch tensor of size (B, D)
        Outputs:
            Matrix of size (B, C).
        """
        # Layer 1
        x_mapped = self.input_to_hidden(x)
        x_relu = F.relu(x_mapped)
        x_relu_dropout = self.dropout(x_relu)

        # Layer 2
        x_mapped = self.hidden_to_hidden(x_relu_dropout)
        x_relu = F.relu(x_mapped)
        x_relu_dropout = self.dropout(x_relu)

        # Output w/o Softmax
        output = self.hidden_to_output(x_relu_dropout)

        # Softmax
        output = F.softmax(output, dim=1)

        return output
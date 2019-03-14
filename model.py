import torch.nn as nn
import torch as torch
import torch.nn.functional as F
torch.manual_seed(0) # set seed for torch RNG

from util import *

'''
    Write a model for gesture classification.
    3.2
'''

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # L of input is 100... feature map becomes L=70 after 3 convolutions

        NUM_CHANNELS = 6

        self.bn1 = nn.BatchNorm1d(NUM_CHANNELS)
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=10, kernel_size=11)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=15, kernel_size=11)
        self.conv3 = nn.Conv1d(in_channels=15, out_channels=20, kernel_size=11)
        self.fc1 = nn.Linear(in_features=20*70, out_features=64)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=26)

    def forward(self, x):

        x = (self.bn1(x))
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = x.view(-1, 20*70)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = (self.fc3(x))
        return x
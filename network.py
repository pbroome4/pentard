import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from collections import deque
import random
import math
from dataclasses import dataclass
from typing import List


# TODO Move to Rust
@dataclass
class TrainingDatum:
    # TODO state: tensor of the current state?
    # mcts_policy: Policy after performing mcts
    # outcome:  Game win/loss/draw
    pass


class TrainingDataset(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, data: List[TrainingDatum]):
        """Save a transition"""
        self.memory.append(data)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Tard(nn.Module):
    """General AlphaZero inspired nueral-network"""
    def __init__(self, in_chan=4):
        """
        """
        super(Tard, self).__init__()
        
        # 1. Shared Feature Extractor (Convolutional Trunk)
        #    Input is shape (batch_size, input_channels, board_width, board_height)
        # Output shape (batch_size, 64, board_width, board_height)
        self.conv1 = nn.Conv2d(in_chan, 64, kernel_size=3, padding=1)
        # Output shape (batch_size, 64, board_width, board_height)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Output shape (batch_size, 64, board_width, board_height)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Batch normalization helps stabilize training
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        # 2. Policy Head (Decides which move to make)
        # Output shape (batch_size, 2, board_width, board_height)
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1) # Reduce channels to 2
        self.policy_bn = nn.BatchNorm2d(2)
        # Output shape (batch_size, board_width*board_height) but previous layer needs flattening
        self.policy_fc = nn.Linear(2 * engine.action_space_size(), engine.action_space_size())

        # 3. Value Head (Evaluates who is winning: -1 to +1)
        # Output shape (batch_size, 1, board_width, board_height)
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)  # Reduce channels to 1
        self.value_bn = nn.BatchNorm2d(1)
        # Output shape (batch_size, board_width*board_height) but previous layer needs flattening
        self.value_fc1 = nn.Linear(1 * engine.action_space_size(), 64)
        # Output Shape (batch_size, 1)
        self.value_fc2 = nn.Linear(engine.action_space_size(), 1)


    def forward(self, x):
        # Trunk forward pass
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))  
        x = F.relu(self.bn3(self.conv3(x)))  

        # Policy Head forward pass
        p = F.relu(self.policy_bn(self.policy_conv(x))) # shape (batch_size, 2, )
        p = p.view(p.size(0), -1)  # Flatten
        policy = F.softmax(self.policy_fc(p), dim=1)  # Outputs probability distribution over moves

        # Value Head forward pass
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # Flatten
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))  # Outputs scalar between -1 and +1

        return policy, value


    def evaluate(self, rust_game_batch):
        """
        Uses the network to evaluate a batch of games
        rust_game_batch: numpy array representation of Tensor to pass through network.
        Returns: List[policy, value]
            For each game state in the batch, return the network policy and value output
        """
        data = None
        with torch.no_grad:
            data = self(rust_game_batch) # Calls self.forward(), and some other behind the scenes stuff
            #TODO maybe modify data to make it rust friendly?
        return data


def train_tard():
    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tard = Tard()
    optimizer = optim.Adam(tard.parameters(), lr=0.001, weight_decay=1e-4)
    dataset = TrainingDataset(10000)


    iteration_n = 0
    while True:
        iteration_n += 1
        print(f"Iteration {iteration_n}")
        tard.eval() # Temporarily turn off training (specifically weight decay)
        with torch.no_grad: # Temporarily turn off training (Specifically gradient tracking)
            # TODO
            # training_data = run_self_play(tard.evaluate, batch_size, N)
            # dataset.append(training_data) # Is thread_safe
            pass


if __name__=="__main__":
    train_tard()





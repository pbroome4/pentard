import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from collections import deque
import random
import math
from dataclasses import dataclass

import engine

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

#class ReplayMemory(object):
#    def __init__(self, capacity):
#        self.memory = deque([], maxlen=capacity)
#
#    def push(self, *args):
#        """Save a transition"""
#        self.memory.append(Transition(*args))
#
#    def sample(self, batch_size):
#        return random.sample(self.memory, batch_size)
#
#    def __len__(self):
#        return len(self.memory)


class Pentard(nn.Module):
    def __init__(self, in_chan=4):
        """
        in_chan should match the to_tensor function of the GameState
        """
        super(Pentard, self).__init__()
        
        # 1. Shared Feature Extractor (Convolutional Trunk)
        self.conv1 = nn.Conv2d(in_chan, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Batch normalization helps stabilize training
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        # 2. Policy Head (Decides which move to make)
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1) # Reduce channels to 2
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * engine.action_space_size(), engine.action_space_size())

        # 3. Value Head (Evaluates who is winning: -1 to +1)
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)  # Reduce channels to 1
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * engine.action_space_size(), 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch_size, input_channels, board_size, board_size)
        
        # Trunk forward pass
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Policy Head forward pass
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  # Flatten
        policy = F.softmax(self.policy_fc(p), dim=1)  # Outputs probability distribution over moves

        # Value Head forward pass
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # Flatten
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))  # Outputs scalar between -1 and +1

        return policy, value


class MCTSNode:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = {}   # map action_index -> child node
        self.n_vists = 0
        self.val_avg = 0
        



def mcts():
    """ Pure monte carlo tree search(MCTS)"""


def mcts_puct():
    """ Uses predictor upper confidence bounds applied to trees (puct) for monte carlo tree search(MCTS) """
    pass



memory = ReplayMemory(10000)


### Training loop
if torch.cuda.is_available() or torch.backends.mps.is_available():
    n_episodes = 10
else:
    n_episodes = 10

for i_episode in range(n_episodes):
    game_state = engine.GameState()
    state_tens = game_state.to_tensor()


for i_episode in range(n_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
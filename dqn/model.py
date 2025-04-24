import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple

Transition = namedtuple('Transition',
                        ('state','action','reward','next_state','done'))

# network
class QNetwork(nn.Module):
    def __init__(self, state_dim=219, action_dim=109):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# buffer
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# agent for deep q-learning
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99):
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = ReplayBuffer()
        self.steps_done = 0

    def select_action(self, state, eps_threshold):
        # ε-greedy
        if random.random() < eps_threshold:
            return random.randrange(self.policy_net.fc3.out_features)
        with torch.no_grad():
            q_vals = self.policy_net(state.unsqueeze(0))
            return q_vals.argmax().item()

    def optimize(self, batch_size):
        if len(self.buffer) < batch_size:
            return
        trans = self.buffer.sample(batch_size)
        states = torch.stack(trans.state)
        actions = torch.tensor(trans.action)
        rewards = torch.tensor(trans.reward)
        next_states = torch.stack(trans.next_state)
        dones = torch.tensor(trans.done, dtype=torch.bool)

        # current Q
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        # target Q
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (~dones)

        loss = F.mse_loss(q_values, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

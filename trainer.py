import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.episode_losses = []

    def train_step(self, state, action, reward, next_state, done):
        state = np.array(state)
        next_state = np.array(next_state)
        action = np.array(action)
        reward = np.array(reward)

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # Predicted Q values with the current state
        pred = self.model(state)

        # Target Q values
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Perform gradient descent
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        # Accumulate loss for the episode
        self.episode_losses.append(loss.item())

    def get_average_loss(self):
        if self.episode_losses:
            avg_loss = sum(self.episode_losses) / len(self.episode_losses)
            self.episode_losses = []  # Reset for the next episode
            return avg_loss
        else:
            return 0

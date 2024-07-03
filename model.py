import torch
import torch.nn as nn
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name))
        self.eval()  # Set the model to evaluation mode

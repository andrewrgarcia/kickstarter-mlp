
import torch.nn as nn

class KickstarterMLP(nn.Module):
    def __init__(self, input_dim):
        super(KickstarterMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),  # Input layer
            nn.ReLU(),                 # Nonlinear activation
            nn.Linear(64, 1)           # Output layer
        )

    def forward(self, x):
        return self.net(x)
# modules/text_projector.py
import torch
import torch.nn as nn

class SmallMLPProjector(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=256):
        super(SmallMLPProjector, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

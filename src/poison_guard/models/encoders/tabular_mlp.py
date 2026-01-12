import torch
import torch.nn as nn
from typing import List

class TabularMLPEncoder(nn.Module):
    """
    MLP Encoder for tabular data.
    Architecture: Input -> Linear -> BN -> ReLU -> Output
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 128):
        super(TabularMLPEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), # Added a second layer to match common encoder patterns or keep it simple? 
            # The prompt said "Input -> Linear -> BN -> ReLU -> output h". 
            # Let's stick closer to the prompt but ensure output_dim is respected.
            # If the prompt implies a single block, it might output hidden_dim.
            # But usually encoders output a fixed embedding size.
            # Let's do: Linear(input, output) -> BN -> ReLU.
            # Wait, "Input -> Linear -> BN -> ReLU -> output h"
            # If I strictly follow:
        )
        
        # Re-reading prompt: "Input -> Linear -> BN -> ReLU -> output h"
        # This implies a single layer or a block.
        # Let's make it a robust encoder with at least one hidden layer if desired, 
        # but for strict compliance:
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

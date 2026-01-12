import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    Projection Head for Contrastive Learning.
    Architecture: Input h -> Linear -> ReLU -> Linear -> Output z
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super(ProjectionHead, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

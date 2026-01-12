from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch
import hashlib
import json
import datetime

@dataclass(frozen=True)  # frozen=True ensures Immutability in memory
class BehavioralFingerprint:
    """
    Immutable specificiation of a dataset's "Normal" geometric behavior.
    """
    dataset_name: str
    version: str
    
    # --- Geometric Fingerprint ---
    # Centroid of the clean data manifold
    mean_embedding: torch.Tensor 
    
    # Shape of the manifold (Covariance diagonal or full matrix if dims low)
    # Storing full covariance can be heavy, so we might store top-k components or diagonal
    covariance_diag: torch.Tensor
    
    # --- Spectral Fingerprint ---
    # Singular values of the clean data batch (Top-K)
    singular_values: torch.Tensor
    # Entropy-based rank (scalar) 
    effective_rank: float
    
    # --- Density Fingerprint ---
    # Distribution of pairwise cosine similarities (Mean, Std)
    pairwise_sim_mean: float
    pairwise_sim_std: float

    # Metadata
    computed_at_timestamp: str
    # Hash of the raw fingerprint data for integrity check
    integrity_hash: str = field(init=False)

    def __post_init__(self):
        # Calculate integrity hash upon creation
        data_str = (
            f"{self.dataset_name}{self.version}"
            f"{self.effective_rank}{self.pairwise_sim_mean}"
        )
        # We bypass frozen dataclass restriction just for this field using object.__setattr__
        object.__setattr__(self, 'integrity_hash', hashlib.sha256(data_str.encode()).hexdigest())

    def save(self, path: str):
        """Save to disk (atomic write)."""
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> 'BehavioralFingerprint':
        return torch.load(path)
        
    def to_summary_dict(self) -> Dict[str, Any]:
        """Reduced dictionary for logging."""
        return {
            "dataset": self.dataset_name,
            "version": self.version,
            "effective_rank": self.effective_rank,
            "isotropy_score": (self.singular_values[-1] / self.singular_values[0]).item() if len(self.singular_values) > 0 else 0.0,
            "pairwise_sim_mean": self.pairwise_sim_mean
        }

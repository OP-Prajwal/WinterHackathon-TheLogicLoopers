from typing import Dict, Any
import torch
import numpy as np
from ..baselines.fingerprint import BehavioralFingerprint

class DriftDetector:
    """
    The Decision Module.
    Compares live active metrics (Geometry/Spectral) against the immutable Baseline.
    
    Outputs standardized anomaly scores.
    """
    def __init__(self, baseline: BehavioralFingerprint):
        self.baseline = baseline
        
        # Drift sensitivity thresholds (Tunable hyperparameters)
        # Ideally these are set by validating on clean data variance
        self.rank_sensitivity = 0.1  # Max allowable % drop in rank
        self.density_sensitivity = 0.1 # Max allowable shift in cosine density

    def detect(self, 
               current_rank: float, 
               current_density: float) -> Dict[str, Any]:
        """
        Compare current batch metrics vs Baseline.
        
        Args:
            current_rank: Smoothed effective rank from SpectralMonitor.
            current_density: Average cosine similarity from GeometryAnalyzer.
            
        Returns:
            Dict containing scores and status.
        """
        # 1. Spectral Drift (Rank Collapse Check)
        # Collapse is when rank DROPS parameters
        # If baseline rank is 60, and current is 10 -> Massive Collapse.
        base_rank = self.baseline.effective_rank
        # Avoid div by zero
        if base_rank == 0: base_rank = 1.0
            
        rank_change = (current_rank - base_rank) / base_rank
        # We care mostly about negative change (Collapse) for poisoning
        # But positive change (Explosion) is also anomalous
        
        # Score: 0 = No Drift, 1 = Max Drift (Normalized roughly)
        rank_anomaly_score = abs(rank_change) / self.rank_sensitivity
        rank_anomaly_score = min(max(rank_anomaly_score, 0.0), 10.0) # Cap at 10

        # 2. Geometric Drift (Density Check)
        # Poisoning often creates tight clusters (Backdoors), increasing density
        base_density = self.baseline.pairwise_sim_mean
        density_change = current_density - base_density
        
        density_anomaly_score = abs(density_change) / self.density_sensitivity
        density_anomaly_score = min(max(density_anomaly_score, 0.0), 10.0)

        # 3. Aggregate Decision
        # Weighted sum or Max
        total_anomaly_score = (rank_anomaly_score + density_anomaly_score) / 2.0
        
        is_anomalous = total_anomaly_score > 1.0
        
        return {
            "is_anomalous": is_anomalous,
            "total_score": total_anomaly_score,
            "rank_score": rank_anomaly_score,
            "density_score": density_anomaly_score,
            "metrics": {
                "rank_delta": rank_change,
                "density_delta": density_change,
                "current_rank": current_rank,
                "current_density": current_density
            }
        }

    def detect_single_sample(self, embedding: torch.Tensor, raw_features: Dict[str, Any]) -> float:
        """
        Detect anomalies in a single sample using geometric distance and heuristics.
        """
        # 1. Cosine Distance to Baseline Mean
        # baseline.mean_embedding should be on the same device
        if self.baseline.mean_embedding.device != embedding.device:
             mean_emb = self.baseline.mean_embedding.to(embedding.device)
        else:
             mean_emb = self.baseline.mean_embedding

        # cosine similarity = (A . B) / (|A| |B|)
        # cosine distance = 1 - cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(embedding.unsqueeze(0), mean_emb.unsqueeze(0))
        cos_dist = 1.0 - cos_sim.item()
        
        # Normalize distance (assuming mostly aligned, dist < 0.5 usually)
        # Score 0 to 1
        distance_score = min(max(cos_dist * 2.0, 0.0), 1.0)
        
        # 2. Heuristics (Domain Knowledge)
        heuristic_penalty = 0.0
        
        # Example Domain Rules (from prompt)
        if raw_features.get('BMI', 0) > 50:
            heuristic_penalty += 0.5
        
        if raw_features.get('MentHlth', 0) > 30:
            heuristic_penalty += 0.5
            
        final_score = distance_score + heuristic_penalty
        return final_score

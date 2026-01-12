import torch
from typing import Dict, Any, Optional
from ..security_engine.geometry import GeometryAnalyzer
from ..security_engine.spectral import SpectralMonitor
from ..security_engine.drift import DriftDetector
from ..baselines.fingerprint import BehavioralFingerprint

class PipelineAuditor:
    """
    The 'Security Guard' that stands beside the trainer.
    
    Responsibilities:
    1. Receive representation batch 'h'
    2. Run RSE analysis (Spectral, Geometry)
    3. Check against Baseline (Drift)
    4. Return verdict (Continue, Alert, Halt)
    """
    def __init__(self, baseline: BehavioralFingerprint, monitor_window: int = 50):
        self.baseline = baseline
        
        # Private RSE Components
        self._geo = GeometryAnalyzer()
        self._spec = SpectralMonitor(window_size=monitor_window)
        self._drift = DriftDetector(baseline=baseline)
        
        self.strict_mode = False # If True, suggests Halting on Alert

    def audit(self, embeddings: torch.Tensor) -> Dict[str, Any]:
        """
        Run the full security check on a batch.
        """
        # 1. Compute Metrics
        # Note: SpectralMonitor returns the SMOOTHED rank (rolling average)
        current_rank = self._spec.update(embeddings)
        current_density = self._geo.compute_density(embeddings)
        
        # 2. Check Drift
        verdict = self._drift.detect(current_rank, current_density)
        
        # 3. Add Control Signals
        verdict['action'] = "CONTINUE"
        
        if verdict['is_anomalous']:
            verdict['action'] = "HALT" if self.strict_mode else "ALERT"
            
        return verdict

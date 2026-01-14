import time
import statistics
from typing import List, Dict, Any

class MetricTracker:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.total_processed = 0
        self.total_poison = 0
        self.total_safe = 0
        self.start_time = time.time()
        
        # Sliding windows for rates
        self.recent_poison_flags: List[int] = [] # 1 for poison, 0 for safe
        self.drift_scores: List[float] = [] # Placeholder for "Effective Rank" or generic drift metric
        
        # Throughput
        self.last_tick = time.time()
        self.rows_since_tick = 0
        self.current_throughput = 0.0

    def update(self, batch_size: int, poison_count: int, safe_count: int, avg_drift_score: float = 0.0):
        self.total_processed += batch_size
        self.total_poison += poison_count
        self.total_safe += safe_count
        
        # Update sliding window for poison rate
        # We'll approximate by adding "poison_count" 1s and "safe_count" 0s
        # To avoid massive lists, we can just store the aggregate if batch is large, 
        # but for simplicity let's just store the rate of this batch as a single data point if we want smoother graphs,
        # OR widely simpler:
        
        current_batch_poison_rate = poison_count / batch_size if batch_size > 0 else 0
        self.recent_poison_flags.append(current_batch_poison_rate)
        if len(self.recent_poison_flags) > self.window_size:
            self.recent_poison_flags.pop(0)

        # Update drift scores
        self.drift_scores.append(avg_drift_score)
        if len(self.drift_scores) > self.window_size:
            self.drift_scores.pop(0)

        # Update Throughput
        now = time.time()
        self.rows_since_tick += batch_size
        if now - self.last_tick >= 1.0:
            self.current_throughput = self.rows_since_tick / (now - self.last_tick)
            self.last_tick = now
            self.rows_since_tick = 0

    def get_stats(self) -> Dict[str, Any]:
        """Returns snapshot of current metrics"""
        
        # Calculate sliding window averages
        avg_poison_rate = statistics.mean(self.recent_poison_flags) if self.recent_poison_flags else 0.0
        avg_drift = statistics.mean(self.drift_scores) if self.drift_scores else 0.0
        
        return {
            "total_processed": self.total_processed,
            "total_poison": self.total_poison,
            "current_throughput": round(self.current_throughput, 1),
            "poison_rate_window": round(avg_poison_rate * 100, 2), # Percentage
            "drift_score_window": round(avg_drift, 4),
            "system_status": "STABLE" if avg_poison_rate < 0.2 else "HIGH_THREAT" if avg_poison_rate > 0.5 else "ELEVATED"
        }

    def reset(self):
        self.__init__(self.window_size)

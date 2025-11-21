"""
Deep Learning Framework for Time-Series Analysis
Core module providing shared functionality across projects
"""

__version__ = "0.1.0"

import os
import json
from pathlib import Path
from collections import Counter
from datetime import datetime

from .data import TimeSeriesDataset, TimeSeriesDataLoader
from .features import FeatureEngine
from .models import LSTMModel, GRUModel, TransformerModel
from .training import Trainer
from .validation import TimeSeriesSplit, WalkForwardSplit


# Usage Analytics
class UsageTracker:
    """Tracks usage of deep-timeseries features across projects"""

    def __init__(self):
        self.stats = Counter()
        self.enabled = os.environ.get("DEEP_TRACK_USAGE", "true").lower() == "true"
        self.stats_file = Path.home() / ".deep-timeseries" / "usage_stats.json"
        self._load_stats()

    def _load_stats(self):
        """Load existing stats from disk"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
                    self.stats = Counter(data.get('features', {}))
            except Exception:
                pass

    def _save_stats(self):
        """Save stats to disk"""
        if not self.enabled:
            return

        try:
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.stats_file, 'w') as f:
                json.dump({
                    'features': dict(self.stats),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception:
            pass

    def track(self, feature):
        """Track usage of a feature"""
        if self.enabled:
            self.stats[feature] += 1
            self._save_stats()

    def get_stats(self):
        """Get usage statistics"""
        return dict(self.stats)

    def clear_stats(self):
        """Clear all usage statistics"""
        self.stats.clear()
        self._save_stats()


_tracker = UsageTracker()

def track_usage(feature):
    """Track usage of a specific feature"""
    _tracker.track(feature)

def get_usage_stats():
    """Get current usage statistics"""
    return _tracker.get_stats()

def clear_usage_stats():
    """Clear all usage statistics"""
    _tracker.clear_stats()

__all__ = [
    "TimeSeriesDataset",
    "TimeSeriesDataLoader",
    "FeatureEngine",
    "LSTMModel",
    "GRUModel",
    "TransformerModel",
    "Trainer",
    "TimeSeriesSplit",
    "WalkForwardSplit",
    "track_usage",
    "get_usage_stats",
    "clear_usage_stats",
]

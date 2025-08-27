"""
Performance analysis datasets (stub implementation)

This module provides stub implementations for performance-related datasets
that were removed during the AnalysisGNN refactoring.
"""

from analysisgnn.data.data_utils import StandardGraphDataset


class MatchGraphDataset(StandardGraphDataset):
    """
    Stub implementation of MatchGraphDataset for compatibility.
    
    This is a placeholder for the removed performance matching functionality.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with minimal configuration."""
        # Call parent with basic parameters
        super().__init__(
            raw_dir=kwargs.get('raw_dir', None),
            force_reload=kwargs.get('force_reload', False),
            verbose=kwargs.get('verbose', True)
        )
        self._data = []
    
    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self._data)
    
    def __getitem__(self, idx):
        """Get an item from the dataset."""
        if idx >= len(self._data):
            raise IndexError("Dataset index out of range")
        return self._data[idx]
    
    def has_cache(self):
        """Check if the dataset has cached data."""
        return False
    
    def process(self):
        """Process the dataset (stub implementation)."""
        pass
    
    def download(self):
        """Download the dataset (stub implementation)."""
        pass

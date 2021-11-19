import pandas as pd
from forte.processors.data_augment.algorithms.eda_processors import (
    RandomDeletionDataAugmentProcessor,
)
from BaseDataAugmentation import BaseDataAugmenter

class EDAAugmentaion(BaseDataAugmenter):
    def __init__(self, 
                dataset, 
                data_column,
                label_column,
                alpha, 
                augment_frac: float = 0.2) -> None:
        super().__init__(dataset, data_column, label_column, augment_frac)
        self.alpha = alpha
    
    def _create_processor(self):
        configs = {
            'alpha': self.alpha
        }
        processor = RandomDeletionDataAugmentProcessor()
        return processor, configs
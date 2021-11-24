import pandas as pd
from forte.processors.data_augment.algorithms.eda_processors import (
    RandomInsertionDataAugmentProcessor,
)
from augmentation.BaseDataAugmentation import BaseDataAugmenter

class EDAInsertion(BaseDataAugmenter):
    def __init__(self, 
                dataset, 
                data_column,
                label_column,
                alpha, 
                augment_frac: float = 0.2) -> None:
        super().__init__(dataset, data_column, label_column, augment_frac)
        self.alpha = alpha
    
    def create_processor(self):
        configs = {
            'alpha': self.alpha
        }
        processor = RandomInsertionDataAugmentProcessor()
        return processor, configs
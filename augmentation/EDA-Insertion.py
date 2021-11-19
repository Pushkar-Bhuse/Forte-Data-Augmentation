import pandas as pd
from forte.processors.data_augment.algorithms.eda_processors import (
    RandomDeletionDataAugmentProcessor,
    RandomInsertionDataAugmentProcessor,
    RandomSwapDataAugmentProcessor,
)
from BaseDataAugmentation import BaseDataAugmenter

class EDAAugmentaion(BaseDataAugmenter):
    def __init__(self, 
                dataset, 
                data_column,
                label_column,
                alpha, 
                swap_frac: float = 0.2, 
                insert_frac: float = 0.2, 
                delete_frac: float = 0.2) -> None:
        super().__init__(dataset, data_column, label_column)
        self.swap_frac = swap_frac
        self.insert_frac = insert_frac
        self.delete_frac = delete_frac
        self.alpha = alpha
    
    def insert_augmentation(dataset: pd.DataFrame)-> pd.DataFrame:
        

    
    def augment(self, dataset: pd.DataFrame) -> pd.DataFrame:


        
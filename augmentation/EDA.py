import pandas as pd
from forte.processors.data_augment.algorithms.eda_processors import (
    RandomDeletionDataAugmentProcessor,
    RandomInsertionDataAugmentProcessor,
    RandomSwapDataAugmentProcessor,
)

class EDAAugmentaion():
    def __init__(self, swap_frac: float = 0.2, insert_frac: float = 0.2, delete_frac: float = 0.2) -> None:
        self.swap_frac = swap_frac
        self.insert_frac = insert_frac
        self.delete_frac = delete_frac
    
    def insert_augmentation(dataset: pd.DataFrame)-> pd.DataFrame:
        

    
    def augment(self, dataset: pd.DataFrame) -> pd.DataFrame:


        
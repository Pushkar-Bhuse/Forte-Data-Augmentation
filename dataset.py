import pandas as pd

class Dataset():
    def __init__(self, path, **kwargs) -> None:
        sep = kwargs.get("sep", ",")
        self.data = pd.read_csv(path, sep=sep)

    def get_dataset(self):
        return self.data
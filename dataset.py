import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

class Dataset():
    def __init__(self, **kwargs) -> None:
        sep = kwargs.get("sep", ",")
        path = kwargs.get("path", "")
        type = kwargs.get("type", "IMDB")
        self.data_column = kwargs.get("data_column", "DATA_COLUMN")
        self.label_column = kwargs.get("label_column", "LABEL_COLUMN")
        if type == "IMDB":
            URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            dataset = tf.keras.utils.get_file(fname="dataset", 
                                    origin=URL,
                                    untar=True,
                                    cache_dir='.',
                                    cache_subdir='')
            self.train = tf.keras.preprocessing.text_dataset_from_directory(
                'dataset/train', batch_size=30000, validation_split=0.2, 
                subset='training', seed=123)
            self.test = tf.keras.preprocessing.text_dataset_from_directory(
                'dataset/train', batch_size=30000, validation_split=0.2, 
                subset='validation', seed=123)
            for i in self.train.take(1):
                train_feat = i[0].numpy()
                train_lab = i[1].numpy()
            self.train = pd.DataFrame([train_feat, train_lab]).T
            self.train.columns = ['DATA_COLUMN', 'LABEL_COLUMN']
            self.train['DATA_COLUMN'] = self.train['DATA_COLUMN'].str.decode("utf-8")
            for j in self.test.take(1):
                test_feat = j[0].numpy()
                test_lab = j[1].numpy()
            self.test = pd.DataFrame([test_feat, test_lab]).T
            self.test.columns = ['DATA_COLUMN', 'LABEL_COLUMN']
            self.test['DATA_COLUMN'] = self.test['DATA_COLUMN'].str.decode("utf-8")
        else:
            dataset = pd.read_csv(path, sep=sep)
            self.train, self.test = train_test_split(dataset, test_size=0.2)
            
    def get_dataset(self):
        return self.train, self.test

    def get_column_names(self):
        return self.data_column, self.label_column
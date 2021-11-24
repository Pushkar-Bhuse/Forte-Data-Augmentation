import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import shutil

class Dataset():
    def __init__(self, **kwargs) -> None:
        sep = kwargs.get("sep", ",")
        path = kwargs.get("path", "")
        type = kwargs.get("type", "IMDB")
        self.data_column = kwargs.get("data_column", "DATA_COLUMN")
        self.label_column = kwargs.get("label_column", "LABEL_COLUMN")
        self.train_test_split = kwargs.get("train_test_split", 0.2)
        if type == "IMDB":
            URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            dataset = tf.keras.utils.get_file(fname="aclImdb_v1.tar.gz", 
                                    origin=URL,
                                    untar=True,
                                    cache_dir='.',
                                    cache_subdir='')

            # The shutil module offers a number of high-level 
            # operations on files and collections of files.
            # Create main directory path ("/aclImdb")
            main_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
            # Create sub directory path ("/aclImdb/train")
            train_dir = os.path.join(main_dir, 'train')
            # Remove unsup folder since this is a supervised learning task
            remove_dir = os.path.join(train_dir, 'unsup')
            shutil.rmtree(remove_dir)
            # View the final train folder
            print("The dataset is stored in {}".format(os.listdir(train_dir)))

            train = tf.keras.preprocessing.text_dataset_from_directory(
                'aclImdb/train', batch_size=30000, validation_split=self.train_test_split, 
                subset='training', seed=123)
            test = tf.keras.preprocessing.text_dataset_from_directory(
                'aclImdb/train', batch_size=30000, validation_split=self.train_test_split, 
                subset='validation', seed=123)

            for i in train.take(1):
                train_feat = i[0].numpy()
                train_lab = i[1].numpy()
            self.train_data = pd.DataFrame([train_feat, train_lab]).T
            self.train_data.columns = [self.data_column, self.label_column]
            self.train_data[self.data_column] = self.train_data[self.data_column].str.decode("utf-8")

            for j in test.take(1):
                test_feat = j[0].numpy()
                test_lab = j[1].numpy()

            self.test_data = pd.DataFrame([test_feat, test_lab]).T
            self.test_data.columns = [self.data_column, self.label_column]
            self.test[self.data_column] = self.test[self.data_column].str.decode("utf-8")
        else:
            dataset = pd.read_csv(path, sep=sep)
            self.train_data, self.test_data = train_test_split(dataset.sample(frac=0.1), test_size=self.train_test_split)
            
    def get_dataset(self):
        return self.train_data, self.test_data

    def get_column_names(self):
        return self.data_column, self.label_column
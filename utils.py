import os
from os.path import exists
import inspect
import pandas as pd
from importlib import import_module

from augmentation.BaseDataAugmentation import BaseDataAugmenter
from classification.BERTClassifier import BERTClassifier

def append_to_csv(history):
    df = pd.DataFrame.from_dict(history, orient="index")
    if exists("aug_output.csv"):
        df.to_csv("aug_output.csv", mode="a", index=True, header=False)
    else:
        df.to_csv("aug_output.csv")

def get_augmentation_processors(ignore_list = ['__init__.py'], 
                                base_directory = 'augmentation',
                                specific_augemtation = []
                                ):
    augmentation_list = []
    for root, dirs, files in os.walk(f"{base_directory}", topdown=False):    
        for name in files:
            if name not in ignore_list:
                augmentation_list.append(name.split(".")[0])


    augmentation_package = base_directory + ".{}"

    final_list = []

    for a_package in augmentation_list:
        # Import the module:
        module = import_module(augmentation_package.format(a_package))
        # Instantiate the object:
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                if (issubclass(obj, BaseDataAugmenter)) and \
                    ("BaseDataAugmenter" not in name) and \
                    ((name in specific_augemtation) or len(specific_augemtation) == 0):
                    
                    final_list.append({
                        "name": name,
                        "augmentation_class": obj
                    })
                    # Perform Task
    return final_list

def train_augmented_data(model, train_data, data_column, label_column, method):
    history = model.train_model(train_data, 0.2, data_column, label_column)
    return {method: history}
import os
from os.path import exists
import inspect
import pandas as pd
from importlib import import_module

from augmentation.BaseDataAugmentation import BaseDataAugmenter
from classification.BERTClassifier import BERTClassifier

def append_to_csv(results_dict):
    history = {results_dict["augmentation_method"]: results_dict["results"]}
    df = pd.DataFrame.from_dict(history, orient="index")
    if exists("aug_output.csv"):
        df.to_csv("aug_output.csv", mode="a", index=True, header=False)
    else:
        df.to_csv("aug_output.csv")

def fetch_augmentation_processors(ignore_files = ['__init__.py'], 
                                ignore_dirs = ['__pycache__'],
                                base_directory = 'augmentation',
                                specific_augemtation = []
                                ):
    augmentation_list = []
    for root, dirs, files in os.walk(f"{base_directory}", topdown=False):    
        if dirs in ignore_dirs:
            continue
        for name in files:
            if name not in ignore_files:
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
    return final_list
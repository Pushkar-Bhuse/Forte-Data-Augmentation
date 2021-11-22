import os
import inspect
from importlib import import_module
from augmentation.BaseDataAugmentation import BaseDataAugmenter


def get_augmentation_processors(ignore_list = ['__init__.py'], base_directory = 'test_folder'):
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
                if issubclass(obj, BaseDataAugmenter) and "BaseDataAugmenter" not in name:
                    final_list.append(obj)
                    # Perform Task
    return final_list

x = get_augmentation_processors()
print("-----")
print(x[0]().create_processor())
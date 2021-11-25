import argparse
import functools
import logging
import os
import torch
import classification
import utils
from dataset import Dataset
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="classification",
    choices=["classification"],
    help="The type of task on which you want to check the performance of data augmentation",
)

parser.add_argument(
    "--model_type",
    type=str,
    default="BERT",
    choices=["BERT"],
    help="The model to use for performing task.",
)

parser.add_argument(
    "--augmentation_method",
    type=str,
    default="ALL",
    choices=["ALL", "EDA-Insertion", "EDA-Deleteion", "EDA-Swapping", "BackTranslation"],
    help="The type of Data Augmentation Method to test.",
)

parser.add_argument(
    "--dataset_type",
    type=str,
    default="IMDB",
    choices=["IMDB"],
    help="The dataset to perform augmentation on."
)

parser.add_argument(
    "--percentage_augmentation",
    type=float,
    default=0.2,
    help="The fraction of the dataset that will be augmented.",
)

parser.add_argument(
    "--alpha",
    type=float,
    default=0.5,
    help="Aplha value in Forte",
)

parser.add_argument(
    "--train_validation_split",
    type=float,
    default=0.2,
    help="Train Validation Split %",
)

parser.add_argument(
    "--max_len",
    type=int,
    default=128,
    help="Max length of sentence",
)

args = parser.parse_args()

# Checking for the existence of a GPU and initialize it
physical_devices = tf.config.list_physical_devices('GPU') 
machine = "cuda" if physical_devices else "cpu"
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
print("The Experiment is curently using: {}".format(machine))


#Initializing model to train on classification task
def initialize_model():
    if args.model_type == "BERT":
        print("**** Initializing the BERT model ****")
        BERT_Model = classification.BERTClassifier.BERTClassifier()
        return BERT_Model

# Dataset to augment
def initialize_dataset():
    if args.dataset_type == "IMDB":
        print("**** Fetching IMDB Dataset ****")
        dataset = Dataset(type = "IMDB")
        return dataset

# Fetching all Forte Processors for data augmentation 
def generate_augmentation_processors(dataset):
    if args.augmentation_method == "ALL":
        augmentation_methods = utils.fetch_augmentation_processors()
        augmentation_processors = []
        train_data, test_data = dataset.get_dataset()
        data_col, label_col = dataset.get_column_names()

        # Adding Un-Augmented Dataset
        augmentation_processors.append({
            "processor": None,
            "name": "No Augmentation"
        })
        for method in augmentation_methods:
            if method["name"] == "BackTranslator":
                augmentation_processors.append({
                    "processor": method['augmentation_class'](
                        dataset = train_data,
                        data_column = data_col,
                        label_column = label_col,
                        augment_frac = args.percentage_augmentation,
                        device = machine
                    ),
                    "name": method['name']
                })
            else:
                augmentation_processors.append({
                    "processor": method['augmentation_class'](
                        dataset = train_data,
                        data_column = data_col,
                        label_column = label_col,
                        alpha = args.alpha,
                        augment_frac = args.percentage_augmentation,
                    ),
                    "name": method["name"]
                })
    return augmentation_processors
    

def main():
    dataset = initialize_dataset()
    model = initialize_model()
    augmentation_processors = generate_augmentation_processors(dataset)
    for processor in augmentation_processors:
        print("Testing for: {}".format(processor['name']))
        if processor['name'] == "No Augmentation":
            training_data = dataset.get_dataset()[0]
        else:
            training_data = processor['processor'].augment_data(args.max_len)

        testing_data = dataset.get_dataset()[1]
        training_results, test_results = model.train_test_model(
            training_data,
            testing_data, 
            args.train_validation_split, 
            dataset.get_column_names()[0],
            dataset.get_column_names()[1],
            args.max_len
        )
        history = training_results
        history["test_loss"] = test_results[0]
        history["test_accuracy"] = test_results[1]
        utils.append_to_csv({
            "augmentation_method": processor['name'],
            "results": history
        })

if __name__ == "__main__":
    main()
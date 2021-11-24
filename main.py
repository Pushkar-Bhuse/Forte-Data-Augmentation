import argparse
import functools
import logging
import os
import torch
import classification
import utils
from dataset import Dataset

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
    default=0.5,
    help="The fraction of the dataset that will be augmented.",
)

parser.add_argument(
    "--alpha",
    type=float,
    default=0.5,
    help="Aplha value in Forte",
)

args = parser.parse_args()

# Checking for the existence of a GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.root.setLevel(logging.INFO)
print("The Experiment is curently using: {}".format(device))


def initialize_model():
    if args.model_type == "BERT":
        print("**** Initializing the BERT model ****")
        BERT_Model = classification.BERTClassifier.BERTClassifier()
        return BERT_Model

def initialize_dataset():
    if args.dataset_type == "IMDB":
        dataset = Dataset(type = "IMDB")
        return dataset

def get_augmentation_processors(dataset):
    if args.augmentation_method == "ALL":
        augmentation_methods = utils.get_augmentation_processors()
        augmentation_processors = []
        train_data, test_data = dataset.get_dataset()
        data_col, label_col = dataset.get_column_names()
        for method in augmentation_methods:
            if method["name"] == "BackTranslator":
                augmentation_processors.append({
                    "processor": method['augmentation_class'](
                        dataset = train_data,
                        data_column = data_col,
                        label_column = label_col,
                        augment_frac = args.percentage_augmentation,
                        device = device.type
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
    augmentation_processors = get_augmentation_processors(dataset)
    for processor in augmentation_processors:
        augmented_data = processor['processor'].augment_data()
        train_results = utils.train_augmented_data(
                            model, 
                            augmented_data,
                            dataset.get_column_names()[0],
                            dataset.get_column_names()[1],
                            processor['name']
                        )
        utils.append_to_csv(train_results)

if __name__ == "__main__":
    main()


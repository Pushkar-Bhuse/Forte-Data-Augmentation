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
    "--model-type",
    type=str,
    default="BERT",
    choices=["BERT"],
    help="The model to use for performing task.",
)

parser.add_argument(
    "--augmentation-method",
    type=str,
    default="all",
    choices=["ALL", "EDA-Insertion", "EDA-Deleteion", "EDA-Swapping", "BackTranslation"],
    help="The type of Data Augmentation Method to test.",
)

parser.add_argument(
    "--dataset-type",
    type=str,
    default="IMDB",
    choices=["IMDB"],
    help="The dataset to perform augmentation on."
)

parser.add_argument(
    "--percentage-augmentation",
    type=float,
    default=0.5,
    help="The fraction of the dataset that will be augmented.",
)

args = parser.parse_args()

# Checking for the existence of a GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.root.setLevel(logging.INFO)
print("The Experiment is curently using: {}".format(device))


def initialize_model():
    if args.model-type == "BERT":
        print("**** Initializing the BERT model ****")
        BERT_Model = classification.BERTClassifier.BERTClassifier()
        return BERT_Model

def initialize_dataset():
    if args.dataset-type == "IMDB":
        dataset = Dataset(type = "IMDB")
        return dataset

def get_augmentation_processors(dataset):
    if args.augmentation-method == "ALL":
        augmentation_methods = utils.get_augmentation_processors()
        augmentation_processors = []
        for method in augmentation_methods:
            if method["name"] == "BackTranslator":
                augmentation_processors.append(
                    method['augmentation_class']()
                )

            augmentation_processors.append(
                method
            )











import argparse
import functools
import logging
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained-model-name",
    type=str,
    default="bert-base-uncased",
    choices=tx.modules.BERTEncoder.available_checkpoints(),
    help="Name of the pre-trained downstream checkpoint to load.",
)
parser.add_argument(
    "--output-dir",
    default="output/",
    help="The output directory where the model checkpoints will be written.",
)
parser.add_argument(
    "--do-train", action="store_true", help="Whether to run training."
)
parser.add_argument(
    "--do-eval", action="store_true", help="Whether to run eval on the dev set."
)
parser.add_argument(
    "--do-test",
    action="store_true",
    help="Whether to run test on the test set.",
)
parser.add_argument(
    "--augmentation-model-name",
    type=str,
    default="bert-base-uncased",
    choices=tx.modules.BERTEncoder.available_checkpoints(),
    help="Name of the pre-trained augmentation model checkpoint to load.",
)
parser.add_argument(
    "--num-aug",
    type=int,
    default=4,
    help="number of augmentation samples when fine-tuning aug model",
)
parser.add_argument(
    "--classifier-pretrain-epoch",
    type=int,
    default=10,
    help="number of epochs to pretrain the classifier",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.root.setLevel(logging.INFO)

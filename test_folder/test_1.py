# import sys
# sys.path.insert(1, '/Users/pushkar_bhuse/Forte-Data-Augmentation/augmentation')
from augmentation.BaseDataAugmentation import BaseDataAugmenter


class Hello(BaseDataAugmenter):
    def __init__(self) -> None:
        print("In Here")

    def create_processor(self):
        print("Processor Created")

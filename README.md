# Forte-Data-Augmentation

This repository introduces Data Augmentation techniques for Natural Language Proecssing Tasks by Forte. The current setup tests the performance of various Data Augmentation Techniques implemented using Forte when employed to train a BERT-based text classifier downstream model. The augmentation methods are used from the <a href="https://github.com/asyml/forte" target="_blank">Forte Library</a>.

<aside class="o-link-list">
  <div class="o-link-list__aside-title">The following Augmentation methods are covered:</div>
  <ul class="o-link-list__item-container" >
    <li><a class="o-link-list__item" href='https://github.com/asyml/forte/blob/master/forte/processors/data_augment/algorithms/eda_processors.py'>EDA</a></li>
    <li><a class="o-link-list__item" href='https://github.com/asyml/forte/blob/master/forte/processors/data_augment/algorithms/back_translation_op.py'>Backtranslation Data Augmentation</a></li>
    <li><a class="o-link-list__item" href='https://github.com/asyml/forte/blob/master/forte/processors/data_augment/algorithms/dictionary_replacement_op.py'>Dictionary Replacement Data Augmentation</a></li>
  </ul>
</aside>

## Installation and Execution

### Requirements
 <ul class="o-link-list__item-container" >
  <li>Python 3.6 or above</li>
 </ul>
 
### Instructions
```
git clone https://github.com/Pushkar-Bhuse/Forte-Data-Augmentation.git
cd Forte-Data-Augmentation
python3 -m pip install -r requirements.txt
python3 main.py
```
The results of the operations will be stored in `aug_output.csv`.

### Adding an augmentation method
Fork the repository and create new file `my_new_augmenter.py` inside the augmentation folder. Next, create a class and make it extend `BaseDataAugmenter.py` using `from augmentation.BaseDataAugmentation import BaseDataAugmenter`.__
Put all processor configs in the `create_processor` method and define the processor in it. This method must return `processor, processor_config`.__
The `__init__` method requires arguments: dataset,  data_column, label_column, augment_frac. Take any additional arguments using kwargs.__
Check out <a href="https://github.com/Pushkar-Bhuse/Forte-Data-Augmentation/blob/main/augmentation/backtranslation.py" target="_blank">Backtranslation</a> for an example.

from forte.processors.data_augment import ReplacementDataAugmentProcessor
from augmentation.BaseDataAugmentation import BaseDataAugmenter

class DictionaryReplacement(BaseDataAugmenter):
    def __init__(self, dataset, data_column, label_column, augment_frac: float = 0.2, **kwargs) -> None:
        super().__init__(dataset, data_column, label_column, augment_frac)
        self.dictionary_class = kwargs.get("dictionary_tuple", (
            "forte.processors.data_augment."
            "algorithms.dictionary.WordnetDictionary"
        ))
        self.src_lang = kwargs.get("src_lang", "eng")
        self.prob = kwargs.get("prob", 1)

    def create_processor(self):
        processor_config = {
            'lang': self.src_lang,
            'prob': self.prob,
            'dictionary_class': self.dictionary_class
        }

        processor_config = {
            'augment_entry': 'ft.onto.base_ontology.Token',
            'other_entry_policy': {
                'kwargs': {
                    'ft.onto.base_ontology.Document': 'auto_align',
                    'ft.onto.base_ontology.Sentence': 'auto_align'
                }
            },
            'type': 'data_augmentation_op',
            'data_aug_op': 'forte.processors.data_augment.algorithms.dictionary_replacement_op.DictionaryReplacementOp',
            'data_aug_op_config': {
                'kwargs': {
                    'lang': self.src_lang,
                    'prob': self.prob,
                    'dictionary_class': self.dictionary_class
                }
            }
        }
        processor = ReplacementDataAugmentProcessor()
        return processor, processor_config
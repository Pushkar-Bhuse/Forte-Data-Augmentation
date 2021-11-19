from forte.processors.data_augment import ReplacementDataAugmentProcessor
from BaseDataAugmentation import BaseDataAugmenter

class BackTranslator(BaseDataAugmenter):
    def __init__(self, dataset, data_column, label_column, augment_frac: float = 0.2, **kwargs) -> None:
        super().__init__(dataset, data_column, label_column, augment_frac)
        self.src_lang = kwargs.get("src_lang", "en")
        self.target_lang = kwargs.get("target_lang", "de")
        self.prob = kwargs.get("prob", 0.6)
        self.cuda = kwargs.get("cuda", "cpu")

    def _create_processor(self):
        processor_config = {
            'augment_entry': 'ft.onto.base_ontology.Token',
            'other_entry_policy': {
                'kwargs': {
                    'ft.onto.base_ontology.Document': 'auto_align',
                    'ft.onto.base_ontology.Sentence': 'auto_align'
                }
            },
            'type': 'data_augmentation_op',
            'data_aug_op': 'forte.processors.data_augment.algorithms.back_translation_op.BackTranslationOp',
            'data_aug_op_config': {
                'kwargs': {
                    'model_to': 'forte.processors.data_augment.algorithms.machine_translator.MarianMachineTranslator',
                    'model_back': 'forte.processors.data_augment.algorithms.machine_translator.MarianMachineTranslator',
                    'src_language': self.src_lang,
                    'tgt_language': self.target_lang,
                    'device': self.cuda,
                    'prob': self.prob
                }
            }
        }
        processor = ReplacementDataAugmentProcessor()
        return processor, processor_config
    
    def augment(self):
        processor, processor_config = self._create_processor()
        return super()._augment_data(processor, processor_config)
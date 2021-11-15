import pandas as pd

from forte.processors.data_augment import ReplacementDataAugmentProcessor
from forte.pipeline import Pipeline
from forte.data.multi_pack import MultiPack, DataPack
from forte.data.readers import MultiPackSentenceReader, StringReader
from forte.data.caster import MultiPackBoxer
from forte.data.selector import AllPackSelector
from forte.processors.misc import WhiteSpaceTokenizer

# import the dataset and preprocessing class

class DataAugment():
    def __init__(self, method, c, **kwargs) -> None:
        src_lang = kwargs.get("src_lang", "en")
        target_lang = kwargs.get("target_lang", "de")
        prob = kwargs.get("prob", 0.6)
        cuda = kwargs.get("cuda", "cpu")
        augment_col_name = kwargs.get("augment_col", "review")
        rating_col_name = kwargs.get("rating_col", "rating")
        # get dataset from class
        dataset = ""
        if method == "backtranslation":
            self.back_translate(dataset, src_lang, target_lang, prob,
                 cuda, augment_col_name, rating_col_name)

    def back_translate(dataset, src_lang, target_lang, prob, cuda, augment_col, rating_col):
        nlp = Pipeline[MultiPack]()
        
        # Configuration for the data augmentation processor.
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
                    'src_language': src_lang,
                    'tgt_language': target_lang,
                    'device': cuda,
                    'prob': prob
                }
            }
        }

        processor = ReplacementDataAugmentProcessor()

        boxer_config = {"pack_name": "input_src"}
        nlp.add(component=MultiPackBoxer(), config=boxer_config)
        nlp.set_reader(reader=StringReader())
        nlp.add(component=WhiteSpaceTokenizer(), selector=AllPackSelector()) 
        nlp.add(component=processor, config=processor_config)
        nlp.initialize()

        augmented_data = []

        for idx, m_pack in enumerate(nlp.process_dataset(dataset[augment_col])):
            augmented_data.append({
                augment_col: m_pack.get_pack("augmented_input_src").text,
                rating_col: dataset[rating_col][idx]
            })

        augmented_data_df = pd.DataFrame(augmented_data)
        return pd.concat([dataset, augmented_data_df], axis=0)
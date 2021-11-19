import pandas as pd
import os

from forte.processors.data_augment import ReplacementDataAugmentProcessor
from forte.pipeline import Pipeline
from forte.data.multi_pack import MultiPack, DataPack
from forte.data.readers import MultiPackSentenceReader, StringReader
from forte.data.caster import MultiPackBoxer
from forte.data.selector import AllPackSelector
from forte.processors.misc import WhiteSpaceTokenizer
from forte.processors.data_augment.algorithms.embedding_similarity_replacement_op import (
    EmbeddingSimilarityReplacementOp,
)

from texar.torch.data import Embedding, load_glove

# import the dataset and preprocessing class
from dataset import Dataset
from preprocessing import denoise_text

class DataAugment():
    def __init__(self, data_path, method, **kwargs) -> None:
        src_lang = kwargs.get("src_lang", "en")
        target_lang = kwargs.get("target_lang", "de")
        prob = kwargs.get("prob", 0.6)
        cuda = kwargs.get("cuda", "cpu")
        augment_col_name = kwargs.get("augment_col", "review")
        rating_col_name = kwargs.get("rating_col", "rating")
        vocab_path = kwargs.get("vocab_path", "")
        embedding_path = kwargs.get("embedding_path", "")
        embedding_dimension = kwargs.get("embedding_dim", 50)
        
        datapath = Dataset(data_path)
        data = datapath.get_dataset()
        data[augment_col_name] = data[augment_col_name].apply(denoise_text)

        if method == "backtranslation":
            self.back_translate(data, src_lang, target_lang, prob,
                 cuda, augment_col_name, rating_col_name)
        elif method == "embedding_similarity":
            self.embedding_similarity(data, vocab_path, embedding_path,
             embedding_dimension, augment_col_name, rating_col_name)

    
    def _initialize_pipeline():
        nlp = Pipeline[MultiPack]()
        boxer_config = {"pack_name": "input_src"}
        nlp.add(component=MultiPackBoxer(), config=boxer_config)
        nlp.set_reader(reader=StringReader())
        nlp.add(component=WhiteSpaceTokenizer(), selector=AllPackSelector())
        return nlp

    
    def _augment_data(nlp, dataset, augment_col, rating_col):
        augmented_data = []
        for idx, m_pack in enumerate(nlp.process_dataset(dataset[augment_col])):
            augmented_data.append({
                augment_col: m_pack.get_pack("augmented_input_src").text,
                rating_col: dataset[rating_col][idx]
            })
        augmented_data_df = pd.DataFrame(augmented_data)
        return pd.concat([dataset, augmented_data_df], axis=0)

    
    def back_translate(self, dataset, src_lang, target_lang, prob, cuda, augment_col, rating_col):
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

        nlp = self._initialize_pipeline()
        nlp.add(component=processor, config=processor_config)
        nlp.initialize()

        return self._augment_data(nlp, dataset, augment_col, rating_col)

    
    def embedding_similarity(self, dataset, vocab_path, embed_path, embed_dim, augment_col, rating_col):
        file_dir_path = os.path.dirname(__file__)
        self.abs_vocab_path = os.path.abspath(
            os.path.join(file_dir_path, *([os.pardir] * 5), vocab_path)
        )
        abs_embed_path = os.path.abspath(
            os.path.join(file_dir_path, *([os.pardir] * 5), embed_path)
        )
        embed_hparams = Embedding.default_hparams()
        embed_hparams["file"] = abs_embed_path
        embed_hparams["dim"] = embed_dim
        embed_hparams["read_fn"] = load_glove
        self.embed_hparams = embed_hparams
        self.esa = EmbeddingSimilarityReplacementOp(
            configs={
                "vocab_path": self.abs_vocab_path,
                "embed_hparams": self.embed_hparams,
                "top_k": 5,
            }
        )

        processor_config = {
            "augment_entry": "ft.onto.base_ontology.Token",
            "other_entry_policy": {
                "type": "",
                "kwargs": {
                    "ft.onto.base_ontology.Document": "auto_align",
                    "ft.onto.base_ontology.Sentence": "auto_align",
                },
            },
            "type": "data_augmentation_op",
            "data_aug_op": "forte.processors.data_augment.algorithms"
            ".embedding_similarity_replacement_op."
            "EmbeddingSimilarityReplacementOp",
            "data_aug_op_config": {
                "type": "",
                "kwargs": {
                    "vocab_path": self.abs_vocab_path,
                    "embed_hparams": self.embed_hparams,
                    "top_k": 1,
                },
            },
            "augment_pack_names": {"kwargs": {"input": "augmented_input"}},
        }

        nlp = self._initialize_pipeline()
        nlp.add(
            component=ReplacementDataAugmentProcessor(), config=processor_config
        )
        nlp.initialize()

        return self._augment_data(nlp, dataset, augment_col, rating_col)
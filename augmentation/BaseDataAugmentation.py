from abc import ABC, abstractmethod
from forte.data.caster import MultiPackBoxer
from forte.data.multi_pack import MultiPack
from forte.data.readers import StringReader
from forte.data.selector import AllPackSelector
from forte.processors.misc import WhiteSpaceTokenizer

from forte.pipeline import Pipeline
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


class BaseDataAugmenter(ABC):

    def __init__(self, dataset, data_column, label_column, augment_frac) -> None:
        self.dataset = dataset
        self.data_column = data_column
        self.label_column = label_column
        self.augment_frac = augment_frac
    
    def _initialize_pipeline(self):
        nlp = Pipeline[MultiPack]()
        boxer_config = {"pack_name": "input_src"}
        nlp.add(component=MultiPackBoxer(), config=boxer_config)
        nlp.set_reader(reader=StringReader())
        nlp.add(component=WhiteSpaceTokenizer(), selector=AllPackSelector())
        return nlp

    def limit_max_length(self, text, max_length):
        words = word_tokenize(text)
        if len(words) < 128:
            return text
        words = words[:max_length]
        return TreebankWordDetokenizer().detokenize(words)
    
    def augment_data(self, max_len):
        augment_processor, augment_configs = self.create_processor()
        pipeline = self._initialize_pipeline()
        pipeline.add(component=augment_processor, config=augment_configs)
        pipeline.initialize()
        self.dataset[self.data_column] = self.dataset[self.data_column].apply(lambda x: self.limit_max_length(x, max_len))
        try:
            augmented_data = []
            for idx, m_pack in enumerate(pipeline.process_dataset(self.dataset[self.data_column].sample(frac = self.augment_frac))):
                print("Currently on Index: {}".format(idx), end="\t")
                augmented_data.append({
                    self.data_column: m_pack.get_pack("augmented_input_src").text,
                    self.label_column: self.dataset[self.label_column][idx]
                })
        except KeyError:
            print("Something went wrong!")
            pass
        except:
            pass
        augmented_data_df = pd.DataFrame(augmented_data)
        return pd.concat([self.dataset, augmented_data_df], axis=0)

    @abstractmethod
    def create_processor(self):
        pass

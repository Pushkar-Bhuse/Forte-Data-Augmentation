from abc import ABC, abstractmethod
from forte.data.caster import MultiPackBoxer
from forte.data.multi_pack import MultiPack
from forte.data.readers import StringReader
from forte.data.selector import AllPackSelector
from forte.processors.misc import WhiteSpaceTokenizer

from forte.pipeline import Pipeline
import pandas as pd

class BaseDataAugmenter(ABC):

    def __init__(self, dataset = "", data_column = "", label_column = "", augment_frac = 0) -> None:
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

    
    def _augment_data(self):
        augment_processor, augment_configs = self._create_processor()
        pipeline = self._initialize_pipeline()
        pipeline.add(component=augment_processor, config=augment_configs)
        pipeline.initialize()

        augmented_data = []
        for idx, m_pack in enumerate(pipeline.process_dataset(self.dataset[self.data_column].sample(frac = self.augment_frac))):
            augmented_data.append({
                self.data_column: m_pack.get_pack("augmented_input_src").text,
                self.label_column: self.dataset[self.label_column][idx]
            })
        augmented_data_df = pd.DataFrame(augmented_data)
        return pd.concat([self.dataset, augmented_data_df], axis=0)

    @abstractmethod
    def create_processor(self):
        pass

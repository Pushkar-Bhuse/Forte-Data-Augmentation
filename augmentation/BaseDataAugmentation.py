from abc import ABC, abstractmethod
from forte.data.caster import MultiPackBoxer
from forte.data.multi_pack import MultiPack
from forte.data.readers import StringReader
from forte.data.selector import AllPackSelector
from forte.processors.misc import WhiteSpaceTokenizer

from forte.pipeline import Pipeline
import pandas as pd

class BaseDataAugmenter(ABC):
    @abstractmethod
    def __init__(self, data_path, **kwargs):
        pass

    
    def _initialize_pipeline(self):
        nlp = Pipeline[MultiPack]()
        boxer_config = {"pack_name": "input_src"}
        nlp.add(component=MultiPackBoxer(), config=boxer_config)
        nlp.set_reader(reader=StringReader())
        nlp.add(component=WhiteSpaceTokenizer(), selector=AllPackSelector())
        return nlp

    
    def _augment_data(self, augment_processor, augment_configs, dataset, data_column, label_column):
        
        pipeline = self._initialize_pipeline()
        pipeline.add(component=augment_processor, config=augment_configs)
        pipeline.initialize()

        augmented_data = []
        for idx, m_pack in enumerate(pipeline.process_dataset(dataset[data_column])):
            augmented_data.append({
                data_column: m_pack.get_pack("augmented_input_src").text,
                label_column: dataset[label_column][idx]
            })
        augmented_data_df = pd.DataFrame(augmented_data)
        return pd.concat([dataset, augmented_data_df], axis=0)

    @abstractmethod
    def _create_augmentation(self, **kwargs):
        pass

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau


class BERTClassifier():
    def __init__(self) -> None:
        self.model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def _convert_data_to_examples(self, train, validation, test, DATA_COLUMN, LABEL_COLUMN): 
        train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                                text_a = x[DATA_COLUMN], 
                                                                text_b = None,
                                                                label = x[LABEL_COLUMN]), axis = 1)

        validation_InputExamples = validation.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                                text_a = x[DATA_COLUMN], 
                                                                text_b = None,
                                                                label = x[LABEL_COLUMN]), axis = 1)
        
        test_InputExamples = test.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                                text_a = x[DATA_COLUMN], 
                                                                text_b = None,
                                                                label = x[LABEL_COLUMN]), axis = 1)

        return train_InputExamples, validation_InputExamples, test_InputExamples

    def _convert_examples_to_tf_dataset(self, examples, max_length=128):
        features = []

        for e in examples:

            input_dict = self.tokenizer.encode_plus(
                e.text_a,
                add_special_tokens=max_length,
                max_length=max_length, # truncates if len(s) > max_length
                return_token_type_ids=True,
                return_attention_mask=True,
                padding="max_length", # pads to the right by default # CHECK THIS for pad_to_max_length
                truncation=True
            )

            input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
                input_dict["token_type_ids"], input_dict['attention_mask'])

            features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
                )
            )

        def gen():
            for f in features:
                yield (
                    {
                        "input_ids": f.input_ids,
                        "attention_mask": f.attention_mask,
                        "token_type_ids": f.token_type_ids,
                    },
                    f.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    def train_test_model(self, 
                    dataset,
                    test_set, 
                    validation_split, 
                    data_column, 
                    label_column,
                    max_length, 
                    batch_size = 32, 
                    epochs = 2) -> dict:
        train, validation = train_test_split(dataset, test_size=validation_split)

        #Finding the max length of data in both train and validation data
        train_max_len = np.max(train[data_column].str.len())
        validation_max_len = np.max(validation[data_column].str.len())

        train_InputExamples, validation_InputExamples, test_InputExamples = self._convert_data_to_examples(train, validation, test_set, data_column, label_column)

        train_data = self._convert_examples_to_tf_dataset(list(train_InputExamples), max_length)
        train_data = train_data.shuffle(100).batch(batch_size).repeat(epochs)

        validation_data = self._convert_examples_to_tf_dataset(list(validation_InputExamples), max_length)
        validation_data = validation_data.batch(batch_size)

        test_data = self._convert_examples_to_tf_dataset(list(test_InputExamples), max_length)
        test_data = test_data.shuffle(100).batch(batch_size)


        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-6, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.01,
                                    patience=5, min_lr=0.001)
        print("Training Model")
        self.history = self.model.fit(train_data, epochs=2, verbose=1, validation_data=validation_data, callbacks=[reduce_lr])
        print("Testing Model")
        test_results = self.model.evaluate(test_data)
        return self.history.history, test_results


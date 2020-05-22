# Runs BERT model
# (Bidirectional) -> looks at both left AND right context of token
# ()


# Why not Word2Vec -> does not take into account different meanings in different context
# (May be worth training)

# ELMo / ULMFiT -> introductory transfer learning (Pre-Trained model + Fine - Tune the last layer)

# Transformers like BERT are much faster to train than LSTM type models such as ELMo

# 3 Embeddings
# Position embeddings (position of word in a sentence)
# Segment embeddings (one sentence to the next)
# Token embeddings

# Dropout (randomly mast ~15-20% of words)
# Sometimes replace with [MASK] token, sometimes replace with random word to prevent overfitting

from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert


class BertLayer(tf.layers.Layer):
    def __init__(self, n_fine_tune_layers=10, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            bert_path,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        trainable_vars = self.bert.variables

        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]

        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        # Add non-trainable weights
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "pooled_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

def create_train_validation_sets(train):
    TOTAL_BATCHES = math.ceil(len(sorted_reviews_labels) / BATCH_SIZE)
    TEST_BATCHES = TOTAL_BATCHES // 10
    batched_dataset.shuffle(TOTAL_BATCHES)
    test_data = batched_dataset.take(TEST_BATCHES)
    train_data = batched_dataset.skip(TEST_BATCHES)

    return train_data, test_data


def preprocess_for_bert():
    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
        trainable=False,
    )
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

    def _tokenize(text):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    tokenized_text = [_tokenize(text) for text in corpus]

    return tokenized_text


def prepare_batch():
    reviews_with_len = [
        [text, y[i], len(text)] for i, text in enumerate(tokenized_text)
    ]

    # sort by length
    reviews_with_len.sort(key=lambda x: x[2])

    sorted_text_labels = [
        (review_lab[0], review_lab[1]) for review_lab in reviews_with_len
    ]

    processed_dataset = tf.data.Dataset.from_generator(
        lambda: sorted_text_labels, output_types=(tf.int32, tf.int32)
    )

    BATCH_SIZE = 32
    batched_dataset = processed_dataset.padded_batch(
        BATCH_SIZE, padded_shapes=((None,), ())
    )


def run_model_train():
    # TODO: Add BERT embedding layer
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    # Instantiate the custom Bert Layer defined above
    bert_output = BertLayer(n_fine_tune_layers=10)(bert_inputs)

    # Build the rest of the classifier
    dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(
        [train_input_ids, train_input_masks, train_segment_ids],
        train_labels,
        validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_labels),
        epochs=1,
        batch_size=32
    )
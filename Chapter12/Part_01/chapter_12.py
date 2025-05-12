import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DatasetPreparer:
    """Downloads, extracts, and preprocesses the IMDB dataset for language modeling."""
    @staticmethod
    def download_and_extract():
        os.system("wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
        os.system("tar -xf aclImdb_v1.tar.gz")

    @staticmethod
    def clean_text(text):
        """Removes HTML tags from the text."""
        return tf.strings.regex_replace(text, "<br />", " ")

    def prepare_dataset(self, directory="aclImdb", batch_size=256):
        """Loads the dataset and cleans the text."""
        dataset = keras.utils.text_dataset_from_directory(directory=directory, label_mode=None, batch_size=batch_size)
        dataset = dataset.map(lambda x: self.clean_text(x))
        return dataset

class TextVectorizer:
    """Handles tokenization and vectorization of text data."""
    def __init__(self, vocab_size=15000, sequence_length=100):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.vectorizer = layers.TextVectorization(
            max_tokens=self.vocab_size,
            output_mode="int",
            output_sequence_length=self.sequence_length
        )

    def adapt(self, dataset):
        """Fits the vectorizer to the dataset."""
        self.vectorizer.adapt(dataset)

    def transform(self, text_batch):
        """Transforms text into integer sequences."""
        vectorized_sequences = self.vectorizer(text_batch)
        x = vectorized_sequences[:, :-1]
        y = vectorized_sequences[:, 1:]
        return x, y

    def get_vectorized_dataset(self, dataset):
        """Returns a dataset with transformed sequences."""
        return dataset.map(lambda x: self.transform(x), num_parallel_calls=4)

class PositionalEmbedding(layers.Layer):
    """Applies token and positional embeddings to the input text."""
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(layers.Layer):
    """Implements a Transformer decoder block with self-attention and feed-forward layers."""
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

    def get_causal_attention_mask(self, inputs):
        """Generates a causal mask to prevent the model from looking ahead."""
        input_shape = tf.shape(inputs)
        sequence_length = input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        return tf.tile(mask[tf.newaxis, :, :], [tf.shape(inputs)[0], 1, 1])

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        padding_mask = mask if mask is None else tf.cast(mask[:, tf.newaxis, :], dtype="int32")
        padding_mask = tf.minimum(padding_mask, causal_mask) if mask is not None else causal_mask

        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=attention_output_1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask
        )
        attention_output_2 = self.layernorm_2(attention_output_1 + attention_output_2)

        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)

class TransformerTextGenerator:
    """Builds and compiles the transformer-based text generation model."""
    @staticmethod
    def build(sequence_length, vocab_size, embed_dim=256, dense_dim=2048, num_heads=2):
        inputs = keras.Input(shape=(None,), dtype="int64")
        x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
        x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, x)
        outputs = layers.Dense(vocab_size, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop")
        return model

class TextGenerator(keras.callbacks.Callback):
    """Generates text at different temperatures at the end of each epoch."""
    def __init__(self, prompt, generate_length, model_input_length, vectorizer, temperatures=(1.0,), print_freq=1):
        self.prompt = prompt
        self.generate_length = generate_length
        self.model_input_length = model_input_length
        self.temperatures = temperatures
        self.print_freq = print_freq
        self.vectorizer = vectorizer
        self.tokens_index = dict(enumerate(self.vectorizer.vectorizer.get_vocabulary()))
        vectorized_prompt = self.vectorizer.vectorizer([prompt])[0].numpy()
        self.prompt_length = np.nonzero(vectorized_prompt == 0)[0][0]

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_freq != 0:
            return

        for temperature in self.temperatures:
            print(f"\n== Generating with temperature {temperature} ==")
            sentence = self.prompt

            for _ in range(self.generate_length):
                tokenized_sentence = self.vectorizer.vectorizer([sentence])
                predictions = self.model(tokenized_sentence)
                next_token = self._sample_next(predictions[0, self.prompt_length - 1, :], temperature)
                sentence += " " + self.tokens_index[next_token]

            print(sentence)

    @staticmethod
    def _sample_next(predictions, temperature=1.0):
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        return np.argmax(np.random.multinomial(1, predictions, 1))


DatasetPreparer.download_and_extract()
dataset_prep = DatasetPreparer()
dataset = dataset_prep.prepare_dataset()

vectorizer = TextVectorizer()
vectorizer.adapt(dataset)
lm_dataset = vectorizer.get_vectorized_dataset(dataset)

model = TransformerTextGenerator.build(sequence_length=100, vocab_size=15000)
text_gen_callback = TextGenerator(prompt="This movie", generate_length=50, model_input_length=100,
                                  vectorizer=vectorizer, temperatures=(0.2, 0.5, 0.7, 1.0, 1.5))

model.fit(lm_dataset, epochs=200, callbacks=[text_gen_callback])
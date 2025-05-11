import os
import pathlib
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DatasetDownloader:
    @staticmethod
    def download_and_extract():
        os.system("curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
        os.system("tar -xf aclImdb_v1.tar.gz")
        os.system("rm -r aclImdb/train/unsup")

class DatasetPreparer:
    def __init__(self, base_dir="aclImdb", batch_size=32):
        self.base_dir = pathlib.Path(base_dir)
        self.train_dir = self.base_dir / "train"
        self.val_dir = self.base_dir / "val"
        self.batch_size = batch_size

    def split_validation_data(self):
        for category in ("neg", "pos"):
            os.makedirs(self.val_dir / category, exist_ok=True)
            files = os.listdir(self.train_dir / category)
            random.Random(1337).shuffle(files)
            num_val_samples = int(0.2 * len(files))
            val_files = files[-num_val_samples:]
            for fname in val_files:
                shutil.move(self.train_dir / category / fname, self.val_dir / category / fname)

    def load_datasets(self):
        train_ds = keras.utils.text_dataset_from_directory(self.train_dir, batch_size=self.batch_size)
        val_ds = keras.utils.text_dataset_from_directory(self.val_dir, batch_size=self.batch_size)
        test_ds = keras.utils.text_dataset_from_directory(self.base_dir / "test", batch_size=self.batch_size)
        return train_ds, val_ds, test_ds

class TextProcessor:
    def __init__(self, max_tokens=20000, max_length=600):
        self.vectorizer = layers.TextVectorization(
            max_tokens=max_tokens, output_mode="int", output_sequence_length=max_length)

    def adapt(self, dataset):
        text_only_dataset = dataset.map(lambda x, y: x)
        self.vectorizer.adapt(text_only_dataset)

    def transform(self, dataset):
        return dataset.map(lambda x, y: (self.vectorizer(x), y), num_parallel_calls=4)

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

class PositionalEmbedding(layers.Layer):
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config

class TransformerSentimentModel:
    @staticmethod
    def build(vocab_size=20000, sequence_length=600, embed_dim=256, num_heads=2, dense_dim=32):
        inputs = keras.Input(shape=(None,), dtype="int64")
        x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
        x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
        return model

class Trainer:
    def __init__(self, model, train_ds, val_ds, model_name):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.model_name = model_name

    def train(self, epochs=20):
        callbacks = [keras.callbacks.ModelCheckpoint(f"{self.model_name}.keras", save_best_only=True)]
        history = self.model.fit(self.train_ds.cache(),
                                 validation_data=self.val_ds.cache(),
                                 epochs=epochs,
                                 callbacks=callbacks)
        return history.history

    def evaluate(self, test_ds):
        best_model = keras.models.load_model(f"{self.model_name}.keras",
                                             custom_objects={"TransformerEncoder": TransformerEncoder,
                                                             "PositionalEmbedding": PositionalEmbedding})
        test_acc = best_model.evaluate(test_ds)[1]
        print(f"Test Accuracy: {test_acc:.3f}")
        return test_acc


DatasetDownloader.download_and_extract()
dataset_prep = DatasetPreparer()
dataset_prep.split_validation_data()
train_ds, val_ds, test_ds = dataset_prep.load_datasets()

processor = TextProcessor()
processor.adapt(train_ds)
train_ds = processor.transform(train_ds)
val_ds = processor.transform(val_ds)
test_ds = processor.transform(test_ds)

model = TransformerSentimentModel.build()
trainer = Trainer(model, train_ds, val_ds, "transformer_encoder")
trainer.train(epochs=20)
trainer.evaluate(test_ds)
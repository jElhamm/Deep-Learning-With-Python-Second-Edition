import os
import shutil
import random
import pathlib
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
    def __init__(self, base_dir="aclImdb"):
        self.base_dir = pathlib.Path(base_dir)
        self.train_dir = self.base_dir / "train"
        self.val_dir = self.base_dir / "val"
        self.batch_size = 32

    def split_validation_data(self):
        for category in ("neg", "pos"):
            os.makedirs(self.val_dir / category, exist_ok=True)
            files = os.listdir(self.train_dir / category)
            random.Random(1337).shuffle(files)
            num_val_samples = int(0.2 * len(files))
            val_files = files[-num_val_samples:]
            for fname in val_files:
                shutil.move(self.train_dir / category / fname,
                            self.val_dir / category / fname)

    def load_datasets(self):
        train_ds = keras.utils.text_dataset_from_directory(self.train_dir, batch_size=self.batch_size)
        val_ds = keras.utils.text_dataset_from_directory(self.val_dir, batch_size=self.batch_size)
        test_ds = keras.utils.text_dataset_from_directory(self.base_dir / "test", batch_size=self.batch_size)
        return train_ds, val_ds, test_ds

class TextProcessor:
    def __init__(self, max_tokens=20000, max_length=600):
        self.max_tokens = max_tokens
        self.vectorizer = layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=max_length
        )

    def adapt(self, dataset):
        text_only_dataset = dataset.map(lambda x, y: x)
        self.vectorizer.adapt(text_only_dataset)

    def transform(self, dataset):
        return dataset.map(lambda x, y: (self.vectorizer(x), y), num_parallel_calls=4)

class EmbeddingLoader:
    def __init__(self, glove_path="glove.6B.100d.txt", max_tokens=20000, embedding_dim=100):
        self.glove_path = glove_path
        self.max_tokens = max_tokens
        self.embedding_dim = embedding_dim
        self.embeddings_index = self._load_glove_embeddings()

    def _load_glove_embeddings(self):
        embeddings_index = {}
        with open(self.glove_path) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                embeddings_index[word] = np.fromstring(coefs, "f", sep=" ")
        print(f"Found {len(embeddings_index)} word vectors.")
        return embeddings_index

    def create_embedding_matrix(self, vocabulary):
        word_index = dict(zip(vocabulary, range(len(vocabulary))))
        embedding_matrix = np.zeros((self.max_tokens, self.embedding_dim))
        for word, i in word_index.items():
            if i < self.max_tokens:
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        return embedding_matrix

class SentimentModel:
    @staticmethod
    def build_lstm_model(max_tokens=20000, embedding_dim=256, use_pretrained_embedding=False, embedding_matrix=None):
        inputs = keras.Input(shape=(None,), dtype="int64")

        if use_pretrained_embedding:
            embedding_layer = layers.Embedding(
                input_dim=max_tokens,
                output_dim=embedding_dim,
                embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                trainable=False,
                mask_zero=True,
            )
        else:
            embedding_layer = layers.Embedding(input_dim=max_tokens, output_dim=embedding_dim, mask_zero=True)

        x = embedding_layer(inputs)
        x = layers.Bidirectional(layers.LSTM(32))(x)
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

    def train(self, epochs=10):
        callbacks = [keras.callbacks.ModelCheckpoint(f"{self.model_name}.keras", save_best_only=True)]
        history = self.model.fit(self.train_ds.cache(),
                                 validation_data=self.val_ds.cache(),
                                 epochs=epochs,
                                 callbacks=callbacks)
        return history.history

    def evaluate(self, test_ds):
        best_model = keras.models.load_model(f"{self.model_name}.keras")
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

embedding_loader = EmbeddingLoader()
embedding_matrix = embedding_loader.create_embedding_matrix(processor.vectorizer.get_vocabulary())

model = SentimentModel.build_lstm_model(use_pretrained_embedding=True, embedding_matrix=embedding_matrix)
trainer = Trainer(model, train_ds, val_ds, "glove_lstm_model")
trainer.train(epochs=10)
trainer.evaluate(test_ds)
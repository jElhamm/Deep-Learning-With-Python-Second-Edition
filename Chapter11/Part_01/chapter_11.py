import os
import re
import shutil
import random
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TextProcessor:
    @staticmethod
    def standardize(text):
        text = text.lower()
        return "".join(char for char in text if char not in string.punctuation)

    @staticmethod
    def tokenize(text):
        text = TextProcessor.standardize(text)
        return text.split()

class TextVectorizer:
    def __init__(self, output_mode="multi_hot", max_tokens=20000, ngrams=1):
        self.vectorizer = keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode=output_mode,
            ngrams=ngrams,
            standardize=self.custom_standardization
        )

    @staticmethod
    def custom_standardization(text):
        text = tf.strings.lower(text)
        return tf.strings.regex_replace(text, f"[{re.escape(string.punctuation)}]", "")

    def adapt(self, dataset):
        text_only_dataset = dataset.map(lambda x, y: x)
        self.vectorizer.adapt(text_only_dataset)

    def transform(self, dataset):
        return dataset.map(lambda x, y: (self.vectorizer(x), y), num_parallel_calls=4)

class DatasetPreparer:
    def __init__(self, base_dir="aclImdb"):
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir, "train")
        self.val_dir = os.path.join(base_dir, "val")
        self.batch_size = 32

    def split_validation_data(self):
        for category in ("neg", "pos"):
            os.makedirs(os.path.join(self.val_dir, category), exist_ok=True)
            files = os.listdir(os.path.join(self.train_dir, category))
            random.Random(1337).shuffle(files)
            num_val_samples = int(0.2 * len(files))
            val_files = files[-num_val_samples:]
            for fname in val_files:
                shutil.move(os.path.join(self.train_dir, category, fname),
                            os.path.join(self.val_dir, category, fname))

    def load_datasets(self):
        train_ds = keras.utils.text_dataset_from_directory(self.train_dir, batch_size=self.batch_size)
        val_ds = keras.utils.text_dataset_from_directory(self.val_dir, batch_size=self.batch_size)
        test_ds = keras.utils.text_dataset_from_directory(os.path.join(self.base_dir, "test"), batch_size=self.batch_size)
        return train_ds, val_ds, test_ds

class SentimentModel:
    @staticmethod
    def build_model(max_tokens=20000, hidden_dim=16):
        inputs = keras.Input(shape=(max_tokens,))
        x = layers.Dense(hidden_dim, activation="relu")(inputs)
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

class Inference:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    def predict_sentiment(self, text):
        raw_text_data = tf.convert_to_tensor([[text]])
        processed_inputs = self.vectorizer.vectorizer(raw_text_data)
        prediction = self.model(processed_inputs)[0].numpy()
        sentiment = "positive" if prediction > 0.5 else "negative"
        confidence = float(prediction * 100)
        print(f"Sentiment: {sentiment} ({confidence:.2f}% confidence)")
        return sentiment, confidence


dataset_prep = DatasetPreparer()
dataset_prep.split_validation_data()
train_ds, val_ds, test_ds = dataset_prep.load_datasets()

vectorizer = TextVectorizer(output_mode="multi_hot", max_tokens=20000, ngrams=1)
vectorizer.adapt(train_ds)
train_ds = vectorizer.transform(train_ds)
val_ds = vectorizer.transform(val_ds)
test_ds = vectorizer.transform(test_ds)

model = SentimentModel.build_model()
trainer = Trainer(model, train_ds, val_ds, "binary_1gram")
trainer.train(epochs=10)
trainer.evaluate(test_ds)

inference_model = keras.models.load_model("binary_1gram.keras")
inference = Inference(inference_model, vectorizer)
inference.predict_sentiment("That was an excellent movie, I loved it.")
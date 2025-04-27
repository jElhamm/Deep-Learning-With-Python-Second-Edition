import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb, reuters, boston_housing


class DataLoader:
    @staticmethod
    def load_imdb():
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
        return train_data, train_labels, test_data, test_labels

    @staticmethod
    def load_reuters():
        (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
        return train_data, train_labels, test_data, test_labels

    @staticmethod
    def load_boston_housing():
        (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
        return train_data, train_targets, test_data, test_targets

class Preprocessor:
    @staticmethod
    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.0
        return results

    @staticmethod
    def to_one_hot(labels, dimension=46):
        return keras.utils.to_categorical(labels, num_classes=dimension)

    @staticmethod
    def normalize_data(train_data, test_data):
        mean = train_data.mean(axis=0)
        std = train_data.std(axis=0)
        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std
        return train_data, test_data

class TextModel:
    def __init__(self, output_dim, loss_function):
        self.model = keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(output_dim, activation="softmax" if output_dim > 1 else "sigmoid")
        ])
        self.model.compile(optimizer="rmsprop", loss=loss_function, metrics=["accuracy"])

    def get_model(self):
        return self.model

class RegressionModel:
    @staticmethod
    def build_model():
        model = keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])
        model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
        return model

class Trainer:
    def __init__(self, model, train_data, train_labels, val_data=None, val_labels=None):
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels

    def train(self, epochs=20, batch_size=512):
        history = self.model.fit(
            self.train_data, self.train_labels,
            epochs=epochs, batch_size=batch_size,
            validation_data=(self.val_data, self.val_labels) if self.val_data is not None else None
        )
        return history.history

class Evaluator:
    @staticmethod
    def evaluate(model, test_data, test_labels):
        results = model.evaluate(test_data, test_labels)
        print(f"Test results: {results}")
        return results

    @staticmethod
    def predict(model, test_data):
        predictions = model.predict(test_data)
        return predictions

class Plotter:
    @staticmethod
    def plot_loss(history):
        epochs = range(1, len(history["loss"]) + 1)
        plt.plot(epochs, history["loss"], "bo", label="Training loss")
        plt.plot(epochs, history["val_loss"], "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_accuracy(history):
        epochs = range(1, len(history["accuracy"]) + 1)
        plt.plot(epochs, history["accuracy"], "bo", label="Training accuracy")
        plt.plot(epochs, history["val_accuracy"], "b", label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

train_data, train_labels, test_data, test_labels = DataLoader.load_imdb()
x_train = Preprocessor.vectorize_sequences(train_data)
x_test = Preprocessor.vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

imdb_model = TextModel(output_dim=1, loss_function="binary_crossentropy").get_model()
trainer = Trainer(imdb_model, x_train[10000:], y_train[10000:], x_train[:10000], y_train[:10000])
history = trainer.train(epochs=4)
Evaluator.evaluate(imdb_model, x_test, y_test)
Plotter.plot_loss(history)
Plotter.plot_accuracy(history)

train_data, train_labels, test_data, test_labels = DataLoader.load_reuters()
x_train = Preprocessor.vectorize_sequences(train_data)
x_test = Preprocessor.vectorize_sequences(test_data)
y_train = Preprocessor.to_one_hot(train_labels)
y_test = Preprocessor.to_one_hot(test_labels)

reuters_model = TextModel(output_dim=46, loss_function="categorical_crossentropy").get_model()
trainer = Trainer(reuters_model, x_train[1000:], y_train[1000:], x_train[:1000], y_train[:1000])
history = trainer.train(epochs=9)
Evaluator.evaluate(reuters_model, x_test, y_test)
Plotter.plot_loss(history)
Plotter.plot_accuracy(history)

train_data, train_targets, test_data, test_targets = DataLoader.load_boston_housing()
train_data, test_data = Preprocessor.normalize_data(train_data, test_data)
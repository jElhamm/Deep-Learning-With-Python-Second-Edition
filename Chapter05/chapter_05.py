import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist, imdb

class DataLoader:
    @staticmethod
    def load_mnist():
        (train_images, train_labels), _ = mnist.load_data()
        train_images = train_images.reshape((60000, 28 * 28)).astype("float32") / 255
        return train_images, train_labels

    @staticmethod
    def load_imdb():
        (train_data, train_labels), _ = imdb.load_data(num_words=10000)
        return train_data, train_labels

    @staticmethod
    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.0
        return results

class ModelBuilder:
    @staticmethod
    def build_dense_model(input_shape, output_units, hidden_units=512, activation="relu", output_activation="softmax"):
        model = keras.Sequential([
            layers.Dense(hidden_units, activation=activation, input_shape=(input_shape,)),
            layers.Dense(output_units, activation=output_activation)
        ])
        return model

    @staticmethod
    def build_imdb_model(hidden_units=16, output_units=1, activation="relu", output_activation="sigmoid"):
        model = keras.Sequential([
            layers.Dense(hidden_units, activation=activation),
            layers.Dense(hidden_units, activation=activation),
            layers.Dense(output_units, activation=output_activation)
        ])
        return model

    @staticmethod
    def build_regularized_model(l1_reg=0.0, l2_reg=0.002):
        model = keras.Sequential([
            layers.Dense(16, kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg), activation="relu"),
            layers.Dense(16, kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg), activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])
        return model

    @staticmethod
    def build_dropout_model():
        model = keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid")
        ])
        return model

class Trainer:
    def __init__(self, model, train_data, train_labels, validation_split=0.2, learning_rate=0.001):
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.validation_split = validation_split
        self.learning_rate = learning_rate

    def compile_model(self, loss_function="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]):
        self.model.compile(optimizer=keras.optimizers.RMSprop(self.learning_rate), loss=loss_function, metrics=metrics)

    def train(self, epochs=10, batch_size=128):
        history = self.model.fit(
            self.train_data, self.train_labels,
            epochs=epochs, batch_size=batch_size,
            validation_split=self.validation_split
        )
        return history.history

class Evaluator:
    @staticmethod
    def evaluate(model, test_data, test_labels):
        results = model.evaluate(test_data, test_labels)
        print(f"Test results: {results}")
        return results

class Plotter:
    @staticmethod
    def plot_history(history, metric="accuracy"):
        epochs = range(1, len(history[metric]) + 1)
        plt.plot(epochs, history[metric], "bo", label=f"Training {metric}")
        plt.plot(epochs, history[f"val_{metric}"], "b", label=f"Validation {metric}")
        plt.title(f"Training and validation {metric}")
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.show()

train_images, train_labels = DataLoader.load_mnist()

noisy_data = np.concatenate([train_images, np.random.random((len(train_images), 784))], axis=1)
zeros_data = np.concatenate([train_images, np.zeros((len(train_images), 784))], axis=1)

model = ModelBuilder.build_dense_model(1568, 10)
trainer = Trainer(model, noisy_data, train_labels)
trainer.compile_model()
history_noise = trainer.train()

model = ModelBuilder.build_dense_model(1568, 10)
trainer = Trainer(model, zeros_data, train_labels)
trainer.compile_model()
history_zeros = trainer.train()

Plotter.plot_history(history_noise, "accuracy")
Plotter.plot_history(history_zeros, "accuracy")

random_labels = train_labels[:]
np.random.shuffle(random_labels)

model = ModelBuilder.build_dense_model(784, 10)
trainer = Trainer(model, train_images, random_labels)
trainer.compile_model()
trainer.train(epochs=100)

for lr in [1.0, 0.01]:
    model = ModelBuilder.build_dense_model(784, 10)
    trainer = Trainer(model, train_images, train_labels, learning_rate=lr)
    trainer.compile_model()
    trainer.train()

small_model = ModelBuilder.build_dense_model(784, 10, hidden_units=10)
trainer = Trainer(small_model, train_images, train_labels)
trainer.compile_model()
history_small = trainer.train(epochs=20)
Plotter.plot_history(history_small, "loss")

large_model = ModelBuilder.build_dense_model(784, 10, hidden_units=96)
trainer = Trainer(large_model, train_images, train_labels)
trainer.compile_model()
history_large = trainer.train(epochs=20)

l2_model = ModelBuilder.build_regularized_model(l2_reg=0.002)
trainer = Trainer(l2_model, train_images, train_labels)
trainer.compile_model()
trainer.train()

dropout_model = ModelBuilder.build_dropout_model()
trainer = Trainer(dropout_model, train_images, train_labels)
trainer.compile_model()
trainer.train()
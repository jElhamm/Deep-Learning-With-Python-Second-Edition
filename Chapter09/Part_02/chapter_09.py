import os
import shutil
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

class DatasetPreparer:
    def __init__(self, dataset_path="train", new_base_dir="cats_vs_dogs_small"):
        self.original_dir = pathlib.Path(dataset_path)
        self.new_base_dir = pathlib.Path(new_base_dir)

    def make_subset(self, subset_name, start_index, end_index):
        for category in ("cat", "dog"):
            dir = self.new_base_dir / subset_name / category
            os.makedirs(dir, exist_ok=True)
            fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
            for fname in fnames:
                shutil.copyfile(src=self.original_dir / fname, dst=dir / fname)

    def prepare_data(self):
        self.make_subset("train", start_index=0, end_index=1000)
        self.make_subset("validation", start_index=1000, end_index=1500)
        self.make_subset("test", start_index=1500, end_index=2500)

class ResidualBlock:
    @staticmethod
    def build_residual_block(x, filters, pooling=False):
        residual = x
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)

        if pooling:
            x = layers.MaxPooling2D(2, padding="same")(x)
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(residual)
        elif filters != residual.shape[-1]:
            residual = layers.Conv2D(filters, 1, padding="same")(residual)

        x = layers.add([x, residual])
        return x

class CNNModel:
    @staticmethod
    def build_model(img_size=(180, 180, 3), use_data_augmentation=True):
        inputs = keras.Input(shape=img_size)

        if use_data_augmentation:
            data_augmentation = keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.2),
            ])
            x = data_augmentation(inputs)
        else:
            x = inputs

        x = layers.Rescaling(1./255)(x)
        x = layers.Conv2D(filters=32, kernel_size=5, use_bias=False)(x)

        # افزودن Residual Blocks
        for size in [32, 64, 128, 256, 512]:
            residual = x

            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            residual = layers.Conv2D(size, 1, strides=2, padding="same", use_bias=False)(residual)
            x = layers.add([x, residual])

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        return keras.Model(inputs=inputs, outputs=outputs)

class Trainer:
    def __init__(self, model, train_data, val_data):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data

    def compile_model(self):
        self.model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    def train(self, epochs=100):
        history = self.model.fit(
            self.train_data,
            epochs=epochs,
            validation_data=self.val_data
        )
        return history.history

class Evaluator:
    @staticmethod
    def evaluate_model(model, test_data):
        results = model.evaluate(test_data)
        print(f"Test Accuracy: {results[1]:.3f}")
        return results

class Plotter:
    @staticmethod
    def plot_training_history(history):
        epochs = range(1, len(history["loss"]) + 1)
        loss = history["loss"]
        val_loss = history["val_loss"]
        accuracy = history["accuracy"]
        val_accuracy = history["val_accuracy"]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, accuracy, "bo", label="Training accuracy")
        plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()

        plt.show()

dataset_preparer = DatasetPreparer()
dataset_preparer.prepare_data()

data_loader = DataLoader()
train_dataset, validation_dataset, test_dataset = data_loader.load_data()

model = CNNModel.build_model()
trainer = Trainer(model, train_dataset, validation_dataset)
trainer.compile_model()
history = trainer.train()

Plotter.plot_training_history(history)
Evaluator.evaluate_model(model, test_dataset)
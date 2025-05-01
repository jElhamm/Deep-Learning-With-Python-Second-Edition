import os
import random
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img, img_to_array, array_to_img


class DatasetDownloader:
    @staticmethod
    def download_dataset():
        os.system("wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz")
        os.system("wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz")
        os.system("tar -xf images.tar.gz")
        os.system("tar -xf annotations.tar.gz")

class DataLoader:
    def __init__(self, img_size=(200, 200)):
        self.input_dir = "images/"
        self.target_dir = "annotations/trimaps/"
        self.img_size = img_size
        self.input_img_paths, self.target_paths = self._load_paths()

    def _load_paths(self):
        input_img_paths = sorted(
            [os.path.join(self.input_dir, fname)
             for fname in os.listdir(self.input_dir)
             if fname.endswith(".jpg")]
        )
        target_paths = sorted(
            [os.path.join(self.target_dir, fname)
             for fname in os.listdir(self.target_dir)
             if fname.endswith(".png") and not fname.startswith(".")]
        )
        return input_img_paths, target_paths

    def display_sample_image(self, index=9):
        plt.axis("off")
        plt.imshow(load_img(self.input_img_paths[index]))
        plt.show()

    def display_sample_mask(self, index=9):
        img = img_to_array(load_img(self.target_paths[index], color_mode="grayscale"))
        normalized_array = (img.astype("uint8") - 1) * 127
        plt.axis("off")
        plt.imshow(normalized_array[:, :, 0])
        plt.show()

    def load_dataset(self):
        num_imgs = len(self.input_img_paths)
        random.Random(1337).shuffle(self.input_img_paths)
        random.Random(1337).shuffle(self.target_paths)

        input_imgs = np.zeros((num_imgs,) + self.img_size + (3,), dtype="float32")
        targets = np.zeros((num_imgs,) + self.img_size + (1,), dtype="uint8")

        for i in range(num_imgs):
            input_imgs[i] = self._path_to_input_image(self.input_img_paths[i])
            targets[i] = self._path_to_target(self.target_paths[i])

        return input_imgs, targets

    def _path_to_input_image(self, path):
        return img_to_array(load_img(path, target_size=self.img_size))

    def _path_to_target(self, path):
        img = img_to_array(load_img(path, target_size=self.img_size, color_mode="grayscale"))
        return img.astype("uint8") - 1

class SegmentationModel:
    @staticmethod
    def build_model(img_size=(200, 200), num_classes=3):
        inputs = keras.Input(shape=img_size + (3,))
        x = layers.Rescaling(1./255)(inputs)

        # Convolutional layers
        x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
        x = layers.Conv2D(256, 3, strides=2, activation="relu", padding="same")(x)
        x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)

        # Transpose Convolution layers (Upsampling)
        x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same", strides=2)(x)
        x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same", strides=2)(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)

        # Output layer
        outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
        return keras.Model(inputs, outputs)

class Trainer:
    def __init__(self, model, train_data, train_labels, val_data, val_labels):
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels

    def compile_model(self):
        self.model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

    def train(self, epochs=50, batch_size=64):
        callbacks = [keras.callbacks.ModelCheckpoint("oxford_segmentation.keras", save_best_only=True)]
        history = self.model.fit(
            self.train_data, self.train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.val_data, self.val_labels),
            callbacks=callbacks
        )
        return history.history

class Evaluator:
    @staticmethod
    def evaluate_model(model, test_image):
        plt.axis("off")
        plt.imshow(array_to_img(test_image))
        plt.show()

        mask = model.predict(np.expand_dims(test_image, 0))[0]
        Evaluator.display_mask(mask)

    @staticmethod
    def display_mask(pred):
        mask = np.argmax(pred, axis=-1)
        mask *= 127
        plt.axis("off")
        plt.imshow(mask)
        plt.show()

class Plotter:
    @staticmethod
    def plot_training_history(history):
        epochs = range(1, len(history["loss"]) + 1)
        loss = history["loss"]
        val_loss = history["val_loss"]

        plt.figure()
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()
        plt.show()

data_loader = DataLoader()
input_imgs, targets = data_loader.load_dataset()

num_val_samples = 1000
train_input_imgs = input_imgs[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_input_imgs = input_imgs[-num_val_samples:]
val_targets = targets[-num_val_samples:]

model = SegmentationModel.build_model()
trainer = Trainer(model, train_input_imgs, train_targets, val_input_imgs, val_targets)
trainer.compile_model()
history = trainer.train()

Plotter.plot_training_history(history)
model = keras.models.load_model("oxford_segmentation.keras")
Evaluator.evaluate_model(model, val_input_imgs[4])
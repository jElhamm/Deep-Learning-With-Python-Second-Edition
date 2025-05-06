import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers


class DatasetDownloader:
    @staticmethod
    def download_dataset():
        os.system("wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip")
        os.system("unzip jena_climate_2009_2016.csv.zip")

class DataLoader:
    def __init__(self, file_path="jena_climate_2009_2016.csv"):
        self.file_path = file_path
        self.raw_data, self.temperature, self.mean, self.std = self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        with open(self.file_path) as f:
            data = f.read()

        lines = data.split("\n")[1:]  # حذف Header
        raw_data = np.zeros((len(lines), 14))  # 14 ویژگی به جز دما
        temperature = np.zeros((len(lines),))

        for i, line in enumerate(lines):
            values = [float(x) for x in line.split(",")[1:]]
            temperature[i] = values[1]  # مقدار دما
            raw_data[i, :] = values[:]

        # نرمال‌سازی داده‌ها
        mean = raw_data[:int(0.5 * len(raw_data))].mean(axis=0)
        std = raw_data[:int(0.5 * len(raw_data))].std(axis=0)
        raw_data = (raw_data - mean) / std

        return raw_data, temperature, mean, std

class TimeSeriesGenerator:
    def __init__(self, raw_data, temperature, sampling_rate=6, sequence_length=120, delay=144):
        self.raw_data = raw_data
        self.temperature = temperature
        self.sampling_rate = sampling_rate
        self.sequence_length = sequence_length
        self.delay = delay
        self.batch_size = 256

    def create_datasets(self, train_split=0.5, val_split=0.25):
        num_train_samples = int(train_split * len(self.raw_data))
        num_val_samples = int(val_split * len(self.raw_data))

        train_dataset = keras.utils.timeseries_dataset_from_array(
            self.raw_data[:-self.delay],
            targets=self.temperature[self.delay:],
            sampling_rate=self.sampling_rate,
            sequence_length=self.sequence_length,
            shuffle=True,
            batch_size=self.batch_size,
            start_index=0,
            end_index=num_train_samples
        )

        val_dataset = keras.utils.timeseries_dataset_from_array(
            self.raw_data[:-self.delay],
            targets=self.temperature[self.delay:],
            sampling_rate=self.sampling_rate,
            sequence_length=self.sequence_length,
            shuffle=True,
            batch_size=self.batch_size,
            start_index=num_train_samples,
            end_index=num_train_samples + num_val_samples
        )

        test_dataset = keras.utils.timeseries_dataset_from_array(
            self.raw_data[:-self.delay],
            targets=self.temperature[self.delay:],
            sampling_rate=self.sampling_rate,
            sequence_length=self.sequence_length,
            shuffle=True,
            batch_size=self.batch_size,
            start_index=num_train_samples + num_val_samples
        )

        return train_dataset, val_dataset, test_dataset

class BaselineEvaluator:
    def __init__(self, dataset, mean, std):
        self.dataset = dataset
        self.mean = mean
        self.std = std

    def evaluate_naive_method(self):
        total_abs_err = 0.
        samples_seen = 0
        for samples, targets in self.dataset:
            preds = samples[:, -1, 1] * self.std[1] + self.mean[1]
            total_abs_err += np.sum(np.abs(preds - targets))
            samples_seen += samples.shape[0]
        return total_abs_err / samples_seen

class ModelBuilder:
    @staticmethod
    def build_dense_model(input_shape):
        inputs = keras.Input(shape=input_shape)
        x = layers.Flatten()(inputs)
        x = layers.Dense(16, activation="relu")(x)
        outputs = layers.Dense(1)(x)
        return keras.Model(inputs, outputs)

    @staticmethod
    def build_cnn_model(input_shape):
        inputs = keras.Input(shape=input_shape)
        x = layers.Conv1D(8, 24, activation="relu")(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(8, 12, activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(8, 6, activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(1)(x)
        return keras.Model(inputs, outputs)

    @staticmethod
    def build_lstm_model(input_shape):
        inputs = keras.Input(shape=input_shape)
        x = layers.LSTM(16)(inputs)
        outputs = layers.Dense(1)(x)
        return keras.Model(inputs, outputs)

class Trainer:
    def __init__(self, model, train_data, val_data, model_name):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.model_name = model_name

    def compile_model(self):
        self.model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

    def train(self, epochs=10):
        callbacks = [keras.callbacks.ModelCheckpoint(f"{self.model_name}.keras", save_best_only=True)]
        history = self.model.fit(self.train_data, epochs=epochs, validation_data=self.val_data, callbacks=callbacks)
        return history.history

class Plotter:
    @staticmethod
    def plot_training_history(history):
        epochs = range(1, len(history["mae"]) + 1)
        plt.figure()
        plt.plot(epochs, history["mae"], "bo", label="Training MAE")
        plt.plot(epochs, history["val_mae"], "b", label="Validation MAE")
        plt.title("Training and Validation MAE")
        plt.legend()
        plt.show()


data_loader = DataLoader()
train_data, val_data, test_data = TimeSeriesGenerator(data_loader.raw_data, data_loader.temperature).create_datasets()

baseline = BaselineEvaluator(val_data, data_loader.mean, data_loader.std)
print(f"Validation MAE (Naïve Method): {baseline.evaluate_naive_method():.2f}")

dense_model = ModelBuilder.build_dense_model(input_shape=(120, 14))
trainer = Trainer(dense_model, train_data, val_data, "jena_dense")
trainer.compile_model()
dense_history = trainer.train()

Plotter.plot_training_history(dense_history)
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers, regularizers


class BaseModel:
    def __init__(self, input_shape, output_units, hidden_units=64, activation="relu", output_activation="softmax"):
        self.model = keras.Sequential([
            layers.Dense(hidden_units, activation=activation, input_shape=(input_shape,)),
            layers.Dense(output_units, activation=output_activation)
        ])

    def build_model(self):
        return self.model

    def summary(self):
        return self.model.summary()

class MultiInputModel:
    def __init__(self, vocabulary_size, num_tags, num_departments):
        title = keras.Input(shape=(vocabulary_size,), name="title")
        text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
        tags = keras.Input(shape=(num_tags,), name="tags")

        features = layers.Concatenate()([title, text_body, tags])
        features = layers.Dense(64, activation="relu")(features)

        priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
        department = layers.Dense(num_departments, activation="softmax", name="department")(features)

        self.model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])

    def build_model(self):
        return self.model

class Trainer:
    def __init__(self, model, train_data, train_labels, validation_split=0.2):
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.validation_split = validation_split

    def compile_model(self, loss_function="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]):
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

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

class CustomerTicketModel(keras.Model):
    def __init__(self, num_departments):
        super().__init__()
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.department_classifier = layers.Dense(num_departments, activation="softmax")

    def call(self, inputs):
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department

class Classifier(keras.Model):
    def __init__(self, num_classes=2):
        super().__init__()
        self.dense = layers.Dense(num_classes, activation="softmax" if num_classes > 2 else "sigmoid")

    def call(self, inputs):
        return self.dense(inputs)

class RootMeanSquaredError(keras.metrics.Metric):
    def __init__(self, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_sum = self.add_weight(name="mse_sum", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros", dtype="int32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        mse = tf.reduce_sum(tf.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        num_samples = tf.shape(y_pred)[0]
        self.total_samples.assign_add(num_samples)

    def result(self):
        return tf.sqrt(self.mse_sum / tf.cast(self.total_samples, tf.float32))

    def reset_state(self):
        self.mse_sum.assign(0.)
        self.total_samples.assign(0)

class CallbacksManager:
    @staticmethod
    def get_callbacks():
        return [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2),
            keras.callbacks.ModelCheckpoint(filepath="checkpoint.keras", monitor="val_loss", save_best_only=True)
        ]

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


num_samples = 1280
vocabulary_size = 10000
num_tags = 100
num_departments = 4

title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

multi_input_model = MultiInputModel(vocabulary_size, num_tags, num_departments).build_model()
multi_input_model.compile(optimizer="rmsprop",
                          loss=["mean_squared_error", "categorical_crossentropy"],
                          metrics=[["mean_absolute_error"], ["accuracy"]])

multi_input_model.fit([title_data, text_body_data, tags_data],
                      [priority_data, department_data], epochs=1)

basic_model = BaseModel(input_shape=3, output_units=10).build_model()
trainer = Trainer(basic_model, title_data, priority_data)
trainer.compile_model(loss_function="mean_squared_error", optimizer="rmsprop")
history = trainer.train(epochs=5)
Plotter.plot_history(history, "accuracy")
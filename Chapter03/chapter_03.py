import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

class TensorOperations:
    @staticmethod
    def create_tensors():
        tensors = {
            "ones": tf.ones(shape=(2, 1)),
            "zeros": tf.zeros(shape=(2, 1)),
            "normal": tf.random.normal(shape=(3, 1), mean=0., stddev=1.),
            "uniform": tf.random.uniform(shape=(3, 1), minval=0., maxval=1.)
        }
        return tensors

    @staticmethod
    def variable_operations():
        v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
        v.assign(tf.ones((3, 1)))
        v[0, 0].assign(3.)
        v.assign_add(tf.ones((3, 1)))
        return v

    @staticmethod
    def math_operations():
        a = tf.ones((2, 2))
        b = tf.square(a)
        c = tf.sqrt(a)
        d = b + c
        e = tf.matmul(a, b)
        e *= d
        return e

class GradientCalculator:
    @staticmethod
    def compute_gradient():
        input_var = tf.Variable(initial_value=3.)
        with tf.GradientTape() as tape:
            result = tf.square(input_var)
        return tape.gradient(result, input_var)

    @staticmethod
    def compute_second_derivative():
        time = tf.Variable(0.)
        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                position =  4.9 * time ** 2
            speed = inner_tape.gradient(position, time)
        acceleration = outer_tape.gradient(speed, time)
        return acceleration

class DataGenerator:
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.inputs, self.targets = self.generate_data()

    def generate_data(self):
        negative_samples = np.random.multivariate_normal(
            mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=self.num_samples
        )
        positive_samples = np.random.multivariate_normal(
            mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=self.num_samples
        )
        inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
        targets = np.vstack((
            np.zeros((self.num_samples, 1), dtype="float32"),
            np.ones((self.num_samples, 1), dtype="float32")
        ))
        return inputs, targets

    def plot_data(self):
        plt.scatter(self.inputs[:, 0], self.inputs[:, 1], c=self.targets[:, 0])
        plt.show()

class LinearModel:
    def __init__(self, input_dim=2, output_dim=1):
        self.W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
        self.b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

    def predict(self, inputs):
        return tf.matmul(inputs, self.W) + self.b

    def loss(self, targets, predictions):
        per_sample_losses = tf.square(targets - predictions)
        return tf.reduce_mean(per_sample_losses)

class Trainer:
    def __init__(self, model, learning_rate=0.1):
        self.model = model
        self.learning_rate = learning_rate

    def training_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            predictions = self.model.predict(inputs)
            loss = self.model.loss(targets, predictions)
        grad_W, grad_b = tape.gradient(loss, [self.model.W, self.model.b])
        self.model.W.assign_sub(grad_W * self.learning_rate)
        self.model.b.assign_sub(grad_b * self.learning_rate)
        return loss

    def train(self, inputs, targets, epochs=40):
        for step in range(epochs):
            loss = self.training_step(inputs, targets)
            print(f"Loss at step {step}: {loss:.4f}")

class CustomDenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units), initializer="random_normal")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros")

    def call(self, inputs):
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation:
            y = self.activation(y)
        return y

class KerasModel:
    def __init__(self):
        self.model = keras.Sequential([
            CustomDenseLayer(32, activation=tf.nn.relu),
            CustomDenseLayer(64, activation=tf.nn.relu),
            CustomDenseLayer(32, activation=tf.nn.relu),
            CustomDenseLayer(10, activation=tf.nn.softmax)
        ])
        self.model.compile(optimizer="rmsprop", loss="mean_squared_error", metrics=["accuracy"])

    def train(self, inputs, targets, epochs=5, batch_size=128):
        history = self.model.fit(inputs, targets, epochs=epochs, batch_size=batch_size)
        return history.history

class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, val_inputs, val_targets, batch_size=128):
        predictions = self.model.predict(val_inputs, batch_size=batch_size)
        print(predictions[:10])

data_gen = DataGenerator()
data_gen.plot_data()

linear_model = LinearModel()
trainer = Trainer(linear_model)
trainer.train(data_gen.inputs, data_gen.targets, epochs=40)

keras_model = KerasModel()
keras_model.train(data_gen.inputs, data_gen.targets, epochs=5)

evaluator = ModelEvaluator(keras_model.model)
evaluator.evaluate(data_gen.inputs, data_gen.targets)
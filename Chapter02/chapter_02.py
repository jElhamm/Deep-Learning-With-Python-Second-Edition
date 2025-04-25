#***********************************************************************************************************************
#                                                                                                                      *
#                                   Deep Learning With Python  -  CHAPTER 2                                            *
#                                                                                                                      *
#       - This code creates a fully organized system for loading, processing, training, and evaluating a               *
#         neural network model on the **MNIST** dataset.                                                               *
#                                                                                                                      *
#       - The `DataLoader` class loads and preprocesses image data, while the `NeuralNetwork` class defines a          *
#         **deep neural network** with two layers. The `Trainer` class handles model training using the                *
#         processed data, whereas the `Evaluator` class assesses model performance and makes predictions.              *
#                                                                                                                      *
#       - To optimize data processing, the `BatchGenerator` class manages **batching**, and the `MatrixOperations`     *
#         class performs matrix operations such as **addition, multiplication, and ReLU activation**.                  *
#                                                                                                                      * 
#       - Additionally, the `GradientComputation` class uses `GradientTape` to compute gradients.                      *
#         This **modular and flexible** structure allows for easy expansion and adaptation for more complex models.    *
#                                                                                                                      *
#************************************************************************************************************************


import math
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers


class DataLoader:
    def __init__(self):
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.load_data()

    def load_data(self):
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        return train_images, train_labels, test_images, test_labels

    def preprocess_data(self):
        self.train_images = self.train_images.reshape((60000, 28 * 28)).astype("float32") / 255
        self.test_images = self.test_images.reshape((10000, 28 * 28)).astype("float32") / 255

    def get_data(self):
        return (self.train_images, self.train_labels), (self.test_images, self.test_labels)

    def show_sample(self, index=0):
        plt.imshow(self.train_images[index].reshape(28, 28), cmap=plt.cm.binary)
        plt.show()
        print(f"Label: {self.train_labels[index]}")

class NeuralNetwork:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
            layers.Dense(512, activation="relu", input_shape=(28*28,)),
            layers.Dense(10, activation="softmax")
        ])
        model.compile(optimizer="rmsprop",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    def get_model(self):
        return self.model

class Trainer:
    def __init__(self, model, train_images, train_labels):
        self.model = model
        self.train_images = train_images
        self.train_labels = train_labels

    def train(self, epochs=5, batch_size=128):
        self.model.fit(self.train_images, self.train_labels, epochs=epochs, batch_size=batch_size)

class Evaluator:
    def __init__(self, model, test_images, test_labels):
        self.model = model
        self.test_images = test_images
        self.test_labels = test_labels

    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
        print(f"Test Accuracy: {test_acc:.2f}")

    def predict(self, index=0):
        test_sample = self.test_images[index:index+1]
        predictions = self.model.predict(test_sample)
        predicted_label = np.argmax(predictions[0])
        confidence = predictions[0][predicted_label]
        print(f"Predicted Label: {predicted_label}, Confidence: {confidence:.2f}")
        return predicted_label

class MatrixOperations:
    @staticmethod
    def naive_relu(x):
        assert len(x.shape) == 2
        x = x.copy()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] = max(x[i, j], 0)
        return x

    @staticmethod
    def naive_add(x, y):
        assert len(x.shape) == 2 and x.shape == y.shape
        x = x.copy()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] += y[i, j]
        return x

    @staticmethod
    def naive_dot(x, y):
        assert len(x.shape) == 1 and len(y.shape) == 1
        assert x.shape[0] == y.shape[0]
        z = 0.
        for i in range(x.shape[0]):
            z += x[i] * y[i]
        return z

class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next_batch(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels

class GradientComputation:
    @staticmethod
    def compute_gradient(x):
        x = tf.Variable(x)
        with tf.GradientTape() as tape:
            y = 2 * x + 3
        grad = tape.gradient(y, x)
        return grad.numpy()

    @staticmethod
    def compute_matrix_gradient(X, W, b):
        X = tf.Variable(X)
        W = tf.Variable(W)
        b = tf.Variable(b)
        with tf.GradientTape() as tape:
            y = tf.matmul(X, W) + b
        grad_W, grad_b = tape.gradient(y, [W, b])
        return grad_W.numpy(), grad_b.numpy()

data_loader = DataLoader()
data_loader.preprocess_data()
(train_images, train_labels), (test_images, test_labels) = data_loader.get_data()

neural_network = NeuralNetwork()
trainer = Trainer(neural_network.get_model(), train_images, train_labels)
trainer.train(epochs=5, batch_size=128)

evaluator = Evaluator(neural_network.get_model(), test_images, test_labels)
evaluator.evaluate()
evaluator.predict(index=0)

matrix_op = MatrixOperations()
x = np.random.random((20, 100))
y = np.random.random((20, 100))
print("Relu applied:", matrix_op.naive_relu(x)[:5])

grad_calc = GradientComputation()
grad_x = grad_calc.compute_gradient(0.)
print("Gradient of y = 2x + 3 w.r.t x:", grad_x)
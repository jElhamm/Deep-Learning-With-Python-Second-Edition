import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers


class Encoder(keras.Model):
    """Defines the encoder network of the Variational Autoencoder (VAE)."""

    def __init__(self, latent_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.conv_layers = keras.Sequential([
            layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"),
            layers.Flatten(),
            layers.Dense(16, activation="relu"),
        ])
        self.z_mean = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")

    def call(self, inputs):
        x = self.conv_layers(inputs)
        return self.z_mean(x), self.z_log_var(x)

class Sampler(layers.Layer):
    """Sampling layer to generate latent variables using reparameterization trick."""

    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Decoder(keras.Model):
    """Defines the decoder network of the Variational Autoencoder (VAE)."""

    def __init__(self, latent_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.dense = layers.Dense(7 * 7 * 64, activation="relu")
        self.reshape = layers.Reshape((7, 7, 64))
        self.conv_transpose_layers = keras.Sequential([
            layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2D(1, 3, activation="sigmoid", padding="same"),
        ])

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        return self.conv_transpose_layers(x)

class VAE(keras.Model):
    """Defines the Variational Autoencoder (VAE) model."""

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

class VAETrainer:
    """Handles training and visualization for the Variational Autoencoder (VAE)."""

    def __init__(self, latent_dim=2, batch_size=128, epochs=30):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.vae = VAE(self.encoder, self.decoder)
        self.vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)

    def load_data(self):
        """Loads the MNIST dataset and preprocesses it."""
        (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
        mnist_digits = np.concatenate([x_train, x_test], axis=0)
        mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
        return mnist_digits

    def train(self):
        """Trains the VAE model."""
        data = self.load_data()
        self.vae.fit(data, epochs=self.epochs, batch_size=self.batch_size)

    def visualize_latent_space(self, n=30, digit_size=28):
        """Generates and visualizes digits from the latent space."""
        figure = np.zeros((digit_size * n, digit_size * n))
        grid_x = np.linspace(-1, 1, n)
        grid_y = np.linspace(-1, 1, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.vae.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[
                    i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size,
                ] = digit

        plt.figure(figsize=(15, 15))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.axis("off")
        plt.imshow(figure, cmap="Greys_r")
        plt.show()


vae_trainer = VAETrainer()
vae_trainer.train()
vae_trainer.visualize_latent_space()
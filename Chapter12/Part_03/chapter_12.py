import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


class NeuralStyleTransfer:
    def __init__(self, base_image_url, style_image_url, img_height=400, style_weight=1e-6, content_weight=2.5e-8, tv_weight=1e-6):
        """
            Initializes the Neural Style Transfer model with given parameters.

            - base_image_url: URL of the base image
            - style_image_url: URL of the style reference image
            - img_height: Height of the processed images
            - style_weight: Weight for style loss
            - content_weight: Weight for content loss
            - tv_weight: Weight for total variation loss
        """
        self.base_image_path = keras.utils.get_file("base.jpg", origin=base_image_url)
        self.style_reference_image_path = keras.utils.get_file("style.jpg", origin=style_image_url)

        original_width, original_height = keras.utils.load_img(self.base_image_path).size
        self.img_height = img_height
        self.img_width = round(original_width * img_height / original_height)

        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight

        self.model = keras.applications.vgg19.VGG19(weights="imagenet", include_top=False)
        self.feature_extractor = self.build_feature_extractor()

        self.style_layer_names = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        self.content_layer_name = "block5_conv2"

    def preprocess_image(self, image_path):
        """
            Prepares an image for use with the VGG19 model.
        """
        img = keras.utils.load_img(image_path, target_size=(self.img_height, self.img_width))
        img = keras.utils.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = keras.applications.vgg19.preprocess_input(img)
        return img

    def deprocess_image(self, img):
        """
            Converts a processed image back to a viewable format.
        """
        img = img.reshape((self.img_height, self.img_width, 3))
        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68
        img = img[:, :, ::-1]
        img = np.clip(img, 0, 255).astype("uint8")
        return img

    def build_feature_extractor(self):
        """
            Builds a feature extractor model from VGG19.
        """
        outputs_dict = {layer.name: layer.output for layer in self.model.layers}
        return keras.Model(inputs=self.model.inputs, outputs=outputs_dict)

    @staticmethod
    def content_loss(base_img, combination_img):
        return tf.reduce_sum(tf.square(combination_img - base_img))

    @staticmethod
    def gram_matrix(x):
        x = tf.transpose(x, (2, 0, 1))
        features = tf.reshape(x, (tf.shape(x)[0], -1))
        return tf.matmul(features, tf.transpose(features))

    def style_loss(self, style_img, combination_img):
        """
            Computes the style loss using Gram matrices.
        """
        S = self.gram_matrix(style_img)
        C = self.gram_matrix(combination_img)
        channels = 3
        size = self.img_height * self.img_width
        return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

    def total_variation_loss(self, x):
        """
            Computes the total variation loss for smoothness.
        """
        a = tf.square(x[:, : self.img_height - 1, : self.img_width - 1, :] - x[:, 1:, : self.img_width - 1, :])
        b = tf.square(x[:, : self.img_height - 1, : self.img_width - 1, :] - x[:, : self.img_height - 1, 1:, :])
        return tf.reduce_sum(tf.pow(a + b, 1.25))

    def compute_loss(self, combination_image, base_image, style_reference_image):
        """
            Computes the total loss for optimization.
        """
        input_tensor = tf.concat([base_image, style_reference_image, combination_image], axis=0)
        features = self.feature_extractor(input_tensor)

        loss = tf.zeros(shape=())
        layer_features = features[self.content_layer_name]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss += self.content_weight * self.content_loss(base_image_features, combination_features)

        for layer_name in self.style_layer_names:
            layer_features = features[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            loss += (self.style_weight / len(self.style_layer_names)) * self.style_loss(style_reference_features, combination_features)

        loss += self.tv_weight * self.total_variation_loss(combination_image)
        return loss

    @tf.function
    def compute_loss_and_grads(self, combination_image, base_image, style_reference_image):
        """
            Computes gradients for optimization.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(combination_image, base_image, style_reference_image)
        grads = tape.gradient(loss, combination_image)
        return loss, grads

    def train(self, iterations=4000, learning_rate=100.0, decay_steps=100, decay_rate=0.96):
        """
            Trains the neural style transfer model.
        """
        optimizer = keras.optimizers.SGD(
            keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate
            )
        )

        base_image = self.preprocess_image(self.base_image_path)
        style_reference_image = self.preprocess_image(self.style_reference_image_path)
        combination_image = tf.Variable(self.preprocess_image(self.base_image_path))

        for i in range(1, iterations + 1):
            loss, grads = self.compute_loss_and_grads(combination_image, base_image, style_reference_image)
            optimizer.apply_gradients([(grads, combination_image)])

            if i % 100 == 0:
                print(f"Iteration {i}: loss={loss:.2f}")
                img = self.deprocess_image(combination_image.numpy())
                fname = f"combination_image_at_iteration_{i}.png"
                keras.utils.save_img(fname, img)

        print("Training complete! Final image saved.")


nst = NeuralStyleTransfer(
    base_image_url="https://img-datasets.s3.amazonaws.com/sf.jpg",
    style_image_url="https://img-datasets.s3.amazonaws.com/starry_night.jpg"
)
nst.train()
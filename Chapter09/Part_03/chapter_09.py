import os
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers


class ModelLoader:
    @staticmethod
    def load_model_from_file(model_path):
        model = keras.models.load_model(model_path)
        model.summary()
        return model

    @staticmethod
    def load_pretrained_xception():
        return keras.applications.xception.Xception(weights="imagenet", include_top=True)

    @staticmethod
    def load_xception_feature_extractor(layer_name):
        model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
        return keras.Model(inputs=model.input, outputs=model.get_layer(name=layer_name).output)

class ImageProcessor:
    @staticmethod
    def load_and_preprocess_image(img_url, target_size):
        img_path = keras.utils.get_file(fname=img_url.split("/")[-1], origin=img_url)
        img = keras.utils.load_img(img_path, target_size=target_size)
        array = keras.utils.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return array

    @staticmethod
    def preprocess_for_xception(img_array):
        return keras.applications.xception.preprocess_input(img_array)

    @staticmethod
    def display_image(img_array):
        plt.axis("off")
        plt.imshow(img_array[0].astype("uint8"))
        plt.show()

class FeatureVisualizer:
    def __init__(self, model):
        self.model = model

    def get_activation_model(self):
        layer_outputs = []
        layer_names = []
        for layer in self.model.layers:
            if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
                layer_outputs.append(layer.output)
                layer_names.append(layer.name)
        return keras.Model(inputs=self.model.input, outputs=layer_outputs), layer_names

    def display_layer_activations(self, activations, layer_names, images_per_row=16):
        for layer_name, layer_activation in zip(layer_names, activations):
            n_features = layer_activation.shape[-1]
            size = layer_activation.shape[1]
            n_cols = n_features // images_per_row
            display_grid = np.zeros(((size + 1) * n_cols - 1,
                                     images_per_row * (size + 1) - 1))

            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_index = col * images_per_row + row
                    if channel_index < n_features:
                        channel_image = layer_activation[0, :, :, channel_index].copy()
                        if channel_image.sum() != 0:
                            channel_image -= channel_image.mean()
                            channel_image /= channel_image.std()
                            channel_image *= 64
                            channel_image += 128
                        channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                        display_grid[
                            col * (size + 1): (col + 1) * size + col,
                            row * (size + 1): (row + 1) * size + row] = channel_image

            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.axis("off")
            plt.imshow(display_grid, aspect="auto", cmap="viridis")
            plt.show()

class FilterPatternGenerator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    @staticmethod
    def compute_loss(image, filter_index, feature_extractor):
        activation = feature_extractor(image)
        filter_activation = activation[:, 2:-2, 2:-2, filter_index]
        return tf.reduce_mean(filter_activation)

    @tf.function
    def gradient_ascent_step(self, image, filter_index, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(image)
            loss = self.compute_loss(image, filter_index, self.feature_extractor)
        grads = tape.gradient(loss, image)
        grads = tf.math.l2_normalize(grads)
        image += learning_rate * grads
        return image

    def generate_filter_pattern(self, filter_index, img_size=(200, 200), iterations=30, learning_rate=10.):
        image = tf.random.uniform(minval=0.4, maxval=0.6, shape=(1, *img_size, 3))
        for _ in range(iterations):
            image = self.gradient_ascent_step(image, filter_index, learning_rate)
        return image[0].numpy()

    @staticmethod
    def deprocess_image(image):
        image -= image.mean()
        image /= image.std()
        image *= 64
        image += 128
        return np.clip(image, 0, 255).astype("uint8")

class GradCAM:
    def __init__(self, model, last_conv_layer_name, classifier_layer_names):
        self.model = model
        self.last_conv_layer_model = keras.Model(inputs=model.inputs, outputs=model.get_layer(last_conv_layer_name).output)
        classifier_input = keras.Input(shape=self.last_conv_layer_model.output.shape[1:])
        x = classifier_input
        for layer_name in classifier_layer_names:
            x = model.get_layer(layer_name)(x)
        self.classifier_model = keras.Model(classifier_input, x)

    def compute_heatmap(self, img_array):
        with tf.GradientTape() as tape:
            last_conv_layer_output = self.last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            preds = self.classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        last_conv_layer_output = last_conv_layer_output.numpy()[0]

        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(last_conv_layer_output, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

    @staticmethod
    def apply_heatmap(img_path, heatmap, alpha=0.4):
        img = keras.utils.load_img(img_path)
        img = keras.utils.img_to_array(img)
        heatmap = np.uint8(255 * heatmap)

        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.utils.array_to_img(superimposed_img)
        return superimposed_img

model = ModelLoader.load_pretrained_xception()

img_array = ImageProcessor.load_and_preprocess_image("https://img-datasets.s3.amazonaws.com/elephant.jpg", target_size=(299, 299))


grad_cam = GradCAM(model, "block14_sepconv2_act", ["avg_pool", "predictions"])
heatmap = grad_cam.compute_heatmap(img_array)
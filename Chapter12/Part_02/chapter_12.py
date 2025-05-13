import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications import inception_v3


class DeepDream:
    def __init__(self, image_path, layer_settings=None):
        self.image_path = image_path
        self.model = inception_v3.InceptionV3(weights="imagenet", include_top=False)

        self.layer_settings = layer_settings or {
            "mixed4": 1.0,
            "mixed5": 1.5,
            "mixed6": 2.0,
            "mixed7": 2.5,
        }

        self.feature_extractor = self.build_feature_extractor()

    def build_feature_extractor(self):
        outputs_dict = {
            layer_name: self.model.get_layer(name=layer_name).output
            for layer_name in self.layer_settings.keys()
        }
        return keras.Model(inputs=self.model.input, outputs=outputs_dict)

    def preprocess_image(self):
        img = keras.utils.load_img(self.image_path)
        img = keras.utils.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = keras.applications.inception_v3.preprocess_input(img)
        return img

    def deprocess_image(self, img):
        img = img.reshape((img.shape[1], img.shape[2], 3))
        img /= 2.0
        img += 0.5
        img *= 255.
        return np.clip(img, 0, 255).astype("uint8")

    def compute_loss(self, input_image):
        features = self.feature_extractor(input_image)
        loss = tf.zeros(shape=())

        for layer_name, coeff in self.layer_settings.items():
            activation = features[layer_name]
            loss += coeff * tf.reduce_mean(tf.square(activation[:, 2:-2, 2:-2, :]))
        return loss

    @tf.function
    def gradient_ascent_step(self, image, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(image)
            loss = self.compute_loss(image)
        grads = tape.gradient(loss, image)
        grads = tf.math.l2_normalize(grads)
        image += learning_rate * grads
        return loss, image

    def gradient_ascent_loop(self, image, iterations, learning_rate, max_loss=None):
        for i in range(iterations):
            loss, image = self.gradient_ascent_step(image, learning_rate)
            if max_loss is not None and loss > max_loss:
                break
            print(f"... Loss value at step {i}: {loss:.2f}")
        return image

    def deep_dream(self, step=20., num_octave=3, octave_scale=1.4, iterations=30, max_loss=15.):
        original_img = self.preprocess_image()
        original_shape = original_img.shape[1:3]

        successive_shapes = [original_shape]
        for i in range(1, num_octave):
            shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
            successive_shapes.append(shape)
        successive_shapes = successive_shapes[::-1]

        shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])
        img = tf.identity(original_img)

        for i, shape in enumerate(successive_shapes):
            print(f"Processing octave {i+1} with shape {shape}")
            img = tf.image.resize(img, shape)
            img = self.gradient_ascent_loop(img, iterations=iterations, learning_rate=step, max_loss=max_loss)

            upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
            same_size_original = tf.image.resize(original_img, shape)
            lost_detail = same_size_original - upscaled_shrunk_original_img
            img += lost_detail
            shrunk_original_img = tf.image.resize(original_img, shape)

        final_image = self.deprocess_image(img.numpy())
        keras.utils.save_img("dream.png", final_image)
        return final_image

    def display_image(self, image):
        plt.axis("off")
        plt.imshow(image)
        plt.show()


image_path = keras.utils.get_file("coast.jpg", origin="https://img-datasets.s3.amazonaws.com/coast.jpg")
deep_dream = DeepDream(image_path)

dream_image = deep_dream.deep_dream()
deep_dream.display_image(dream_image)
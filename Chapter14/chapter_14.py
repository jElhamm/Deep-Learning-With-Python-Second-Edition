import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DenseModel:
    """Class for creating fully connected (Dense) models for different tasks."""
    def __init__(self, input_shape, num_units=32):
        self.input_shape = input_shape
        self.num_units = num_units

    def build_binary_classification_model(self):
        inputs = keras.Input(shape=(self.input_shape,))
        x = layers.Dense(self.num_units, activation="relu")(inputs)
        x = layers.Dense(self.num_units, activation="relu")(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="rmsprop", loss="binary_crossentropy")
        return model

    def build_multiclass_classification_model(self, num_classes):
        inputs = keras.Input(shape=(self.input_shape,))
        x = layers.Dense(self.num_units, activation="relu")(inputs)
        x = layers.Dense(self.num_units, activation="relu")(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
        return model

    def build_multilabel_classification_model(self, num_classes):
        inputs = keras.Input(shape=(self.input_shape,))
        x = layers.Dense(self.num_units, activation="relu")(inputs)
        x = layers.Dense(self.num_units, activation="relu")(x)
        outputs = layers.Dense(num_classes, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="rmsprop", loss="binary_crossentropy")
        return model

    def build_regression_model(self, num_values):
        inputs = keras.Input(shape=(self.input_shape,))
        x = layers.Dense(self.num_units, activation="relu")(inputs)
        x = layers.Dense(self.num_units, activation="relu")(x)
        outputs = layers.Dense(num_values)(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="rmsprop", loss="mse")
        return model

class CNNModel:
    """Class for creating CNN-based models."""
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.SeparableConv2D(32, 3, activation="relu")(inputs)
        x = layers.SeparableConv2D(64, 3, activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.SeparableConv2D(64, 3, activation="relu")(x)
        x = layers.SeparableConv2D(128, 3, activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.SeparableConv2D(64, 3, activation="relu")(x)
        x = layers.SeparableConv2D(128, 3, activation="relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
        return model

class LSTMModel:
    """Class for creating LSTM-based models for sequence processing."""
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_single_layer_lstm(self):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.LSTM(32)(inputs)
        outputs = layers.Dense(self.num_classes, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="rmsprop", loss="binary_crossentropy")
        return model

    def build_multi_layer_lstm(self):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.LSTM(32, return_sequences=True)(inputs)
        x = layers.LSTM(32, return_sequences=True)(x)
        x = layers.LSTM(32)(x)
        outputs = layers.Dense(self.num_classes, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="rmsprop", loss="binary_crossentropy")
        return model

class TransformerModel:
    """Class for creating Transformer-based models."""
    def __init__(self, sequence_length, vocab_size, embed_dim, dense_dim, num_heads):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

    def build_encoder_decoder_model(self):
        encoder_inputs = keras.Input(shape=(self.sequence_length,), dtype="int64")
        x = PositionalEmbedding(self.sequence_length, self.vocab_size, self.embed_dim)(encoder_inputs)
        encoder_outputs = TransformerEncoder(self.embed_dim, self.dense_dim, self.num_heads)(x)

        decoder_inputs = keras.Input(shape=(None,), dtype="int64")
        x = PositionalEmbedding(self.sequence_length, self.vocab_size, self.embed_dim)(decoder_inputs)
        x = TransformerDecoder(self.embed_dim, self.dense_dim, self.num_heads)(x, encoder_outputs)
        decoder_outputs = layers.Dense(self.vocab_size, activation="softmax")(x)

        transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        transformer.compile(optimizer="rmsprop", loss="categorical_crossentropy")
        return transformer

    def build_text_classification_model(self):
        inputs = keras.Input(shape=(self.sequence_length,), dtype="int64")
        x = PositionalEmbedding(self.sequence_length, self.vocab_size, self.embed_dim)(inputs)
        x = TransformerEncoder(self.embed_dim, self.dense_dim, self.num_heads)(x)
        x = layers.GlobalMaxPooling1D()(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="rmsprop", loss="binary_crossentropy")
        return model


if __name__ == "__main__":
    num_input_features = 20
    num_classes = 5
    num_values = 1
    height, width, channels = 32, 32, 3
    num_timesteps, num_features = 10, 15
    sequence_length, vocab_size, embed_dim, dense_dim, num_heads = 100, 20000, 64, 256, 8

    dense_model = DenseModel(num_input_features)
    binary_model = dense_model.build_binary_classification_model()
    multi_class_model = dense_model.build_multiclass_classification_model(num_classes)
    regression_model = dense_model.build_regression_model(num_values)

    cnn_model = CNNModel((height, width, channels), num_classes).build_model()
    lstm_model = LSTMModel((num_timesteps, num_features), num_classes)
    single_lstm = lstm_model.build_single_layer_lstm()
    multi_lstm = lstm_model.build_multi_layer_lstm()

    transformer = TransformerModel(sequence_length, vocab_size, embed_dim, dense_dim, num_heads)
    transformer_text_model = transformer.build_text_classification_model()
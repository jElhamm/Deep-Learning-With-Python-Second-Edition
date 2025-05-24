import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers


class SimpleMLP(kt.HyperModel):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, hp):
        units = hp.Int(name="units", min_value=16, max_value=64, step=16)
        optimizer = hp.Choice(name="optimizer", values=["rmsprop", "adam"])

        model = keras.Sequential([
            layers.Dense(units, activation="relu"),
            layers.Dense(self.num_classes, activation="softmax")
        ])

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

class HyperparameterTuner:
    def __init__(self, hypermodel, max_trials=100, executions_per_trial=2, directory="mnist_kt_test"):
        self.tuner = kt.BayesianOptimization(
            hypermodel,
            objective="val_accuracy",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory=directory,
            overwrite=True
        )

    def search(self, x_train, y_train, x_val, y_val):
        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)]

        self.tuner.search(
            x_train, y_train,
            batch_size=128,
            epochs=100,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=2
        )

    def get_best_hyperparameters(self, top_n=4):
        return self.tuner.get_best_hyperparameters(top_n)

    def get_best_models(self, top_n=4):
        return self.tuner.get_best_models(top_n)

class ModelTrainer:
    def __init__(self, x_train_full, y_train_full, x_train, y_train, x_val, y_val):
        self.x_train_full = x_train_full
        self.y_train_full = y_train_full
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def get_best_epoch(self, model_builder, hp):
        model = model_builder.build(hp)
        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)]

        history = model.fit(
            self.x_train, self.y_train,
            validation_data=(self.x_val, self.y_val),
            epochs=100,
            batch_size=128,
            callbacks=callbacks
        )

        val_loss_per_epoch = history.history["val_loss"]
        best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
        print(f"Best epoch: {best_epoch}")
        return best_epoch

    def get_best_trained_model(self, model_builder, hp):
        """آموزش مدل با بهترین تعداد epoch."""
        best_epoch = self.get_best_epoch(model_builder, hp)
        model = model_builder.build(hp)

        model.fit(
            self.x_train_full, self.y_train_full,
            batch_size=128,
            epochs=int(best_epoch * 1.2)
        )
        return model

def prepare_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 28 * 28)).astype("float32") / 255
    x_test = x_test.reshape((-1, 28 * 28)).astype("float32") / 255

    x_train_full = x_train[:]
    y_train_full = y_train[:]

    num_val_samples = 10000
    x_train, x_val = x_train[:-num_val_samples], x_train[-num_val_samples:]
    y_train, y_val = y_train[:-num_val_samples], y_train[-num_val_samples:]

    return x_train_full, y_train_full, x_train, y_train, x_val, y_val, x_test, y_test



keras.mixed_precision.set_global_policy("mixed_float16")
x_train_full, y_train_full, x_train, y_train, x_val, y_val, x_test, y_test = prepare_data()

hypermodel = SimpleMLP(num_classes=10)
tuner = HyperparameterTuner(hypermodel)
tuner.search(x_train, y_train, x_val, y_val)

best_hps = tuner.get_best_hyperparameters(top_n=4)
trainer = ModelTrainer(x_train_full, y_train_full, x_train, y_train, x_val, y_val)

best_models = []
for hp in best_hps:
    model = trainer.get_best_trained_model(hypermodel, hp)
    model.evaluate(x_test, y_test)
    best_models.append(model)

best_models = tuner.get_best_models(top_n=4)
import sys

import argparse

import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
import keras

model_path = "src/lab1/mnist_model.keras"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train or load a model to recognize handwritten digits."
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to an image file to recognize a digit from.",
    )
    args = parser.parse_args()

    try:
        model = keras.models.load_model(model_path)
        print("Model loaded from disk.")
    except ValueError:
        print("No saved model found. Training a new model.")

        mnist = keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = keras.models.Sequential(
            [
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(10, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            x_train, y_train, epochs=5, validation_data=(x_test, y_test)
        )
        model.evaluate(x_test, y_test)

        plt.plot(history.history["accuracy"], label="accuracy")
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        model.save(model_path)
        print("Model saved to disk.")

    if args.image:
        img = keras.utils.load_img(
            args.image, color_mode="grayscale", target_size=(28, 28)
        )
        img_array = keras.utils.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        predicted_digit = tf.argmax(predictions[0]).numpy()
        print(f"The predicted digit is: {predicted_digit}")

        plt.imshow(img_array[0], cmap="gray")
        plt.title(f"Predicted Digit: {predicted_digit}")
        plt.axis("off")
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())

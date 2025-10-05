import sys

import argparse

import matplotlib.pyplot as plt
import numpy as np
import keras

model_path = "src/lab1/mnist_model.keras"


def main() -> int:
    parser = argparse.ArgumentParser()
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
        try:
            img = keras.utils.load_img(
                args.image, color_mode="grayscale", target_size=(28, 28)
            )
            img_array = keras.utils.img_to_array(img)

            img_array = img_array / 255.0  # Normalize
            img_array = img_array.reshape(1, 28, 28)  # Reshape for model input

            # Make prediction
            predictions = model.predict(img_array)
            predicted_digit = np.argmax(predictions[0])
            confidence = predictions[0][predicted_digit] * 100

            print(f"\nPredicted digit: {predicted_digit}")
            print(f"Confidence: {confidence:.2f}%")

            # Show all probabilities
            print("\nProbabilities for each digit:")
            for i, prob in enumerate(predictions[0]):
                print(f"  {i}: {prob * 100:.2f}%")

            # Display the image
            plt.imshow(img_array.reshape(28, 28), cmap="gray")
            plt.title(f"Predicted: {predicted_digit} ({confidence:.2f}%)")
            plt.axis("off")
            plt.show()

        except FileNotFoundError:
            print(f"Error: Image file '{args.image}' not found.")
            return 1
        except Exception as e:
            print(f"Error processing image: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

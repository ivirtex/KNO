from datetime import datetime
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import keras

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
SEED = 42


def load_wine_data():
    # Path to data file
    data_path = Path(__file__).parent / "wine" / "wine.data"

    # Column names according to wine.names
    column_names = [
        "Class",
        "Alcohol",
        "Malic_acid",
        "Ash",
        "Alcalinity_of_ash",
        "Magnesium",
        "Total_phenols",
        "Flavanoids",
        "Nonflavanoid_phenols",
        "Proanthocyanins",
        "Color_intensity",
        "Hue",
        "OD280_OD315",
        "Proline",
    ]

    # Load data
    df = pd.read_csv(data_path, names=column_names)
    print("Original dataset:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")

    # Shuffle dataset
    df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print("\n" + "=" * 50)
    print("Shuffled dataset:")
    print(df_shuffled.head(10))

    # One-hot encoding for 'Class' column
    df_one_hot = pd.get_dummies(df_shuffled, columns=["Class"], prefix="Class")
    print("\n" + "=" * 50)
    print("Dataset with one-hot encoding (pd.get_dummies):")
    print(df_one_hot.head(10))
    print(f"\nShape after encoding: {df_one_hot.shape}")

    X = df_one_hot.drop(columns=["Class_1", "Class_2", "Class_3"]).values
    y = df_one_hot[["Class_1", "Class_2", "Class_3"]].values

    print(f"\nFeatures shape (X): {X.shape}")
    print(f"Labels shape (y): {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=np.argmax(y, axis=1)
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Create and adapt normalizer
    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(X_train)

    return X_train, X_test, y_train, y_test, normalizer


def create_simple_model(normalizer):
    model = keras.Sequential(
        [
            keras.layers.InputLayer(shape=(13,), name="input_layer"),
            normalizer,
            keras.layers.Dense(16, activation="relu", name="hidden_layer"),
            keras.layers.Dense(3, activation="softmax", name="output_layer"),
        ],
        name="simple_model",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    return model


def create_complex_model(normalizer):
    model = keras.Sequential(
        [
            keras.layers.InputLayer(shape=(13,), name="input_layer"),
            normalizer,
            keras.layers.Dense(128, activation="relu", name="hidden_layer_1"),
            keras.layers.BatchNormalization(name="batch_norm_1"),
            keras.layers.Dropout(0.3, name="dropout_1"),
            keras.layers.Dense(64, activation="relu", name="hidden_layer_2"),
            keras.layers.BatchNormalization(name="batch_norm_2"),
            keras.layers.Dropout(0.2, name="dropout_2"),
            keras.layers.Dense(32, activation="relu", name="hidden_layer_3"),
            keras.layers.Dropout(0.1, name="dropout_3"),
            keras.layers.Dense(3, activation="softmax", name="output_layer"),
        ],
        name="complex_model",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    return model


def get_callbacks(log_dir):
    return [
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
        ),
        # # Early stopping - stops training when val_loss stops improving
        # keras.callbacks.EarlyStopping(
        #     monitor="val_loss",
        #     patience=20,
        #     restore_best_weights=True,
        #     verbose=1,
        # ),
        # # Reduce learning rate when val_loss doesn't improve
        # keras.callbacks.ReduceLROnPlateau(
        #     monitor="val_loss",
        #     factor=0.5,
        #     patience=10,
        #     min_lr=1e-7,
        #     verbose=1,
        # ),
    ]


def train_model(model, X_train, y_train, log_dir):
    callbacks = get_callbacks(log_dir)

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
    )

    return history


def evaluate_model(model, X_test, y_test):
    print(f"\nEvaluation of {model.name} on test set ({X_test.shape[0]} samples)")

    test_results = model.evaluate(X_test, y_test)
    test_loss = test_results[0]
    test_accuracy = test_results[1]
    test_auc = test_results[2]

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Test AUC: {test_auc:.4f}")

    # Sample predictions
    print("\nSample predictions:")

    y_pred = model.predict(X_test[:5], verbose=0)

    for i in range(5):
        true_class = np.argmax(y_test[i]) + 1
        pred_class = np.argmax(y_pred[i]) + 1
        confidence = np.max(y_pred[i]) * 100

        print(f"\nSample {i + 1}:")
        print(f"  True class: {true_class}")
        print(f"  Predicted: {pred_class} (confidence: {confidence:.1f}%)")
        print(
            f"  Probabilities: Class1={y_pred[i][0]:.3f}, Class2={y_pred[i][1]:.3f}, Class3={y_pred[i][2]:.3f}"
        )

    return {
        "loss": test_loss,
        "accuracy": test_accuracy,
        "auc": test_auc,
    }


def main() -> int:
    X_train, X_test, y_train, y_test, normalizer = load_wine_data()

    # Simple
    simple_log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S-simple")
    simple_model = create_simple_model(normalizer)
    simple_model.summary()

    train_model(simple_model, X_train, y_train, simple_log_dir)

    # Complex
    complex_log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S-complex")
    complex_model = create_complex_model(normalizer)
    complex_model.summary()

    train_model(complex_model, X_train, y_train, complex_log_dir)

    # Evaluation
    simple_results = evaluate_model(simple_model, X_test, y_test)
    complex_results = evaluate_model(complex_model, X_test, y_test)

    print(
        f"\nSimple model accuracy:  {simple_results['accuracy']:.4f} ({simple_results['accuracy'] * 100:.2f}%)"
    )
    print(
        f"Complex model accuracy: {complex_results['accuracy']:.4f} ({complex_results['accuracy'] * 100:.2f}%)"
    )

    # Save best model
    if simple_results["accuracy"] > complex_results["accuracy"]:
        accuracy_diff = simple_results["accuracy"] - complex_results["accuracy"]
        print(f"\nSimple model is better by {accuracy_diff * 100:.2f}%")
        simple_model.save("best_model.keras")
    elif complex_results["accuracy"] > simple_results["accuracy"]:
        accuracy_diff = complex_results["accuracy"] - simple_results["accuracy"]
        print(f"\nComplex model is better by {accuracy_diff * 100:.2f}%")
        complex_model.save("best_model.keras")
    else:
        print("\n= Both models have equal accuracy")

    return 0


if __name__ == "__main__":
    sys.exit(main())

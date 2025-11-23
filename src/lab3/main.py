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


def main() -> int:
    # Ścieżka do pliku z danymi
    data_path = Path(__file__).parent / "wine" / "wine.data"

    # Nazwy kolumn zgodnie z wine.names
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

    # Wczytanie danych
    df = pd.read_csv(data_path, names=column_names)
    print("Oryginalny zbiór danych:")
    print(df.head(10))
    print(f"\nKształt: {df.shape}")

    # Potasowanie zbioru danych
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("\n" + "=" * 50)
    print("Potasowany zbiór danych:")
    print(df_shuffled.head(10))

    # Kodowanie one-hot dla kolumny 'Class'
    df_one_hot = pd.get_dummies(df_shuffled, columns=["Class"], prefix="Class")
    print("\n" + "=" * 50)
    print("Zbiór z kodowaniem one-hot (pd.get_dummies):")
    print(df_one_hot.head(10))
    print(f"\nKształt po kodowaniu: {df_one_hot.shape}")

    X = df_one_hot.drop(columns=["Class_1", "Class_2", "Class_3"]).values
    y = df_one_hot[["Class_1", "Class_2", "Class_3"]].values

    print(f"\nKształt X (cechy): {X.shape}")
    print(f"Kształt y (etykiety): {y.shape}")

    # Podział danych
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )

    print(f"\nTrain set: {X_train.shape[0]} próbek")
    print(f"Test set: {X_test.shape[0]} próbek")

    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(X_train)

    callbacks = [
        # # Early stopping - zatrzyma trening gdy val_loss przestanie się poprawiać
        # keras.callbacks.EarlyStopping(
        #     monitor="val_loss",
        #     patience=20,
        #     restore_best_weights=True,
        #     verbose=1,
        # ),
        # # Zmniejsz learning rate gdy val_loss się nie poprawia
        # keras.callbacks.ReduceLROnPlateau(
        #     monitor="val_loss",
        #     factor=0.5,
        #     patience=10,
        #     min_lr=1e-7,
        #     verbose=1,
        # ),
    ]

    # Prosty model
    log_dir_simple = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S-simple")

    simple_model: keras.Sequential = keras.Sequential(
        [
            # Warstwa wejściowa
            keras.layers.InputLayer(shape=(13,), name="input_layer"),
            # Normalizacja
            normalizer,
            # Warstwa ukryta
            keras.layers.Dense(16, activation="relu", name="hidden_layer"),
            # Warstwa wyjściowa
            keras.layers.Dense(3, activation="softmax", name="output_layer"),
        ],
        name="simple_model",
    )
    simple_model.summary()
    simple_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    # Trenowanie modelu
    simple_model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks
        + [
            keras.callbacks.TensorBoard(
                log_dir=log_dir_simple,
                histogram_freq=1,
            ),
        ],
    )

    # Ewaluacja
    simple_accuracy = evaluate_model(simple_model, X_test, y_test)

    # Bardziej złożony model
    log_dir_complex = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S-complex")

    complex_model: keras.Sequential = keras.Sequential(
        [
            # Warstwa wejściowa
            keras.layers.InputLayer(shape=(13,), name="input_layer"),
            # Normalizacja
            normalizer,
            # Pierwsza warstwa ukryta
            keras.layers.Dense(128, activation="relu", name="hidden_layer_1"),
            keras.layers.BatchNormalization(name="batch_norm_1"),
            keras.layers.Dropout(0.3, name="dropout_1"),
            # Druga warstwa ukryta
            keras.layers.Dense(64, activation="relu", name="hidden_layer_2"),
            keras.layers.BatchNormalization(name="batch_norm_2"),
            keras.layers.Dropout(0.2, name="dropout_2"),
            # Trzecia warstwa ukryta
            keras.layers.Dense(32, activation="relu", name="hidden_layer_3"),
            keras.layers.Dropout(0.1, name="dropout_3"),
            # Warstwa wyjściowa
            keras.layers.Dense(3, activation="softmax", name="output_layer"),
        ],
        name="complex_model",
    )

    complex_model.summary()

    complex_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    # Trenowanie modelu
    complex_model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks
        + [
            keras.callbacks.TensorBoard(
                log_dir=log_dir_complex,
                histogram_freq=1,
            ),
        ],
    )

    # Ewaluacja
    complex_accuracy = evaluate_model(complex_model, X_test, y_test)

    # Porównanie modeli
    print(
        f"\nSimple model accuracy:  {simple_accuracy:.4f} ({simple_accuracy * 100:.2f}%)"
    )
    print(
        f"Complex model accuracy: {complex_accuracy:.4f} ({complex_accuracy * 100:.2f}%)"
    )

    # Zapisanie wag lepszego modelu
    if simple_accuracy > complex_accuracy:
        print(
            f"\nSimple model is better by {(simple_accuracy - complex_accuracy) * 100:.2f}%"
        )

        simple_model.save("best_model.keras")

    elif complex_accuracy > simple_accuracy:
        print(
            f"\nComplex model is better by {(complex_accuracy - simple_accuracy) * 100:.2f}%"
        )

        complex_model.save("best_model.keras")
    else:
        print("\n= Both models have equal accuracy")

    return 0


def evaluate_model(
    model: keras.Sequential, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Evaluate model on test set and print sample predictions.

    Returns:
        Test AUC (float between 0 and 1) for model comparison.
    """
    print(f"\nEvaluation of {model.name} on test set ({X_test.shape[0]} samples):")

    test_results = model.evaluate(X_test, y_test)
    test_loss = test_results[0]
    test_accuracy = test_results[1]
    test_auc = test_results[2]

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Test AUC: {test_auc:.4f}")

    # Predykcje na kilku próbkach
    print("\nSample predictions:")

    y_pred = model.predict(X_test[:5])

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

    return test_auc


if __name__ == "__main__":
    sys.exit(main())

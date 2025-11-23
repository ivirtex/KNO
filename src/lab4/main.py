import sys
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import keras
import keras_tuner as kt

EPOCHS = 100
BATCH_SIZE = 16
SEED = 42

# Best model from lab3
BASELINE_TEST_ACCURACY = 0.9444
BASELINE_TEST_LOSS = 0.2211
BASELINE_TEST_AUC = 0.9952


def load_wine_data():
    # Path to data file
    data_path = Path(__file__).parent.parent / "lab3" / "wine" / "wine.data"

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
    print(f"Dataset shape: {df.shape}")

    # Shuffle dataset
    df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # One-hot encoding for 'Class' column
    df_one_hot = pd.get_dummies(df_shuffled, columns=["Class"], prefix="Class")

    X = df_one_hot.drop(columns=["Class_1", "Class_2", "Class_3"]).values
    y = df_one_hot[["Class_1", "Class_2", "Class_3"]].values

    print(f"Features shape (X): {X.shape}")
    print(f"Labels shape (y): {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=np.argmax(y, axis=1)
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(X_train)

    return X_train, X_test, y_train, y_test, normalizer


def create_model(normalizer, learning_rate, hidden_units_1, activation):
    model = keras.Sequential(
        [
            keras.layers.InputLayer(shape=(13,), name="input_layer"),
            normalizer,
            keras.layers.Dense(
                hidden_units_1, activation=activation, name="hidden_layer_1"
            ),
            keras.layers.Dense(3, activation="softmax", name="output_layer"),
        ],
        name="wine_classifier",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    return model


def build_model_with_hp(hp, normalizer):
    learning_rate = hp.Float(
        "learning_rate",
        min_value=1e-4,
        max_value=5e-3,
        sampling="log",
    )

    hidden_units_1 = hp.Int(
        "hidden_units_1",
        min_value=16,
        max_value=64,
        step=16,
    )

    activation = hp.Choice(
        "activation",
        values=["relu", "tanh", "sigmoid"],
    )

    return create_model(
        normalizer=normalizer,
        learning_rate=learning_rate,
        hidden_units_1=hidden_units_1,
        activation=activation,
    )


def evaluate_model(model, X_test, y_test):
    print(f"\nEvaluation of {model.name} on test set ({X_test.shape[0]} samples)")

    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss = test_results[0]
    test_accuracy = test_results[1]
    test_auc = test_results[2]

    return test_loss, test_accuracy, test_auc


def plot_confusion_matrix(model, X_test, y_test, save_path="confusion_matrix.png"):
    # Get predictions
    y_pred = model.predict(X_test, verbose=0)

    # Convert one-hot encoded labels to class indices
    y_true_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Class 1", "Class 2", "Class 3"]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, values_format="d")
    plt.title("Confusion Matrix - Wine Classification")
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nConfusion matrix saved to: {save_path}")

    plt.close()

    return cm


def tune_hyperparameters(X_train, y_train, normalizer):
    tuner = kt.Hyperband(
        hypermodel=lambda hp: build_model_with_hp(hp, normalizer),
        objective="val_loss",
    )

    tuner.search_space_summary()

    tuner.search(
        X_train,
        y_train,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=1,
    )

    tuner.results_summary()

    best_hps = tuner.get_best_hyperparameters()[0]

    return best_hps


def main() -> int:
    X_train, X_test, y_train, y_test, normalizer = load_wine_data()

    print("\n1. Tuning hyperparameters...")
    best_hps = tune_hyperparameters(X_train, y_train, normalizer)

    best_model = create_model(
        normalizer=normalizer,
        learning_rate=best_hps.get("learning_rate"),
        hidden_units_1=best_hps.get("hidden_units_1"),
        activation=best_hps.get("activation"),
    )

    print("\n2. Training optimized model...")
    best_model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
    )

    best_model.summary()
    print(
        f"Best hyperparameters: "
        f"learning_rate={best_hps.get('learning_rate')}, "
        f"hidden_units_1={best_hps.get('hidden_units_1')}, "
        f"activation={best_hps.get('activation')}"
    )

    print("\n3. Evaluating optimized model...")
    test_loss, test_accuracy, test_auc = evaluate_model(best_model, X_test, y_test)

    print("\n4. Generating confusion matrix...")
    cm = plot_confusion_matrix(best_model, X_test, y_test, "confusion_matrix.png")

    print(f"\n{'Metric':<20} {'Baseline':<15} {'Tuned':<15} {'Improvement'}")
    print("-" * 60)

    loss_improvement = BASELINE_TEST_LOSS - test_loss
    accuracy_improvement = test_accuracy - BASELINE_TEST_ACCURACY
    auc_improvement = test_auc - BASELINE_TEST_AUC

    print(
        f"{'Test Loss':<20} "
        f"{BASELINE_TEST_LOSS:>6.4f}          "
        f"{test_loss:>6.4f}          "
        f"{loss_improvement:>+.4f}"
    )
    print(
        f"{'Test Accuracy':<20} "
        f"{BASELINE_TEST_ACCURACY:>6.4f} ({BASELINE_TEST_ACCURACY * 100:>5.2f}%)  "
        f"{test_accuracy:>6.4f} ({test_accuracy * 100:>5.2f}%)  "
        f"{accuracy_improvement:>+.4f} ({accuracy_improvement * 100:>+6.2f}%)"
    )
    print(
        f"{'Test AUC':<20} "
        f"{BASELINE_TEST_AUC:>6.4f}          "
        f"{test_auc:>6.4f}          "
        f"{auc_improvement:>+.4f}"
    )

    best_model_path = "best_tuned_model.keras"
    best_model.save(best_model_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

import sys
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import keras_tuner as kt

EPOCHS = 100
BATCH_SIZE = 16
SEED = 42

# Complex model baseline from lab3
BASELINE = {
    "test_accuracy": 0.9444,
    "test_auc": 0.9952,
    "test_loss": 0.2211,
}


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

    # Create and adapt normalization layer
    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(X_train)

    return X_train, X_test, y_train, y_test, normalizer


def create_model(normalizer, learning_rate, hidden_units_1, dropout_rate):
    model = keras.Sequential(
        [
            keras.layers.InputLayer(shape=(13,), name="input_layer"),
            normalizer,
            keras.layers.Dense(
                hidden_units_1, activation="relu", name="hidden_layer_1"
            ),
            keras.layers.Dropout(dropout_rate, name="dropout_1"),
            keras.layers.Dense(32, activation="relu", name="hidden_layer_2"),
            keras.layers.Dropout(dropout_rate / 2, name="dropout_2"),
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
        max_value=128,
        step=32,
    )

    dropout_rate = hp.Float(
        "dropout_rate",
        min_value=0.0,
        max_value=0.4,
        step=0.1,
    )

    return create_model(
        normalizer=normalizer,
        learning_rate=learning_rate,
        hidden_units_1=hidden_units_1,
        dropout_rate=dropout_rate,
    )


def evaluate_model(model, X_test, y_test):
    print(f"\nEvaluation of {model.name} on test set ({X_test.shape[0]} samples)")

    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss = test_results[0]
    test_accuracy = test_results[1]
    test_auc = test_results[2]

    return {
        "loss": test_loss,
        "accuracy": test_accuracy,
        "auc": test_auc,
    }


def tune_hyperparameters(X_train, y_train, normalizer):
    tuner = kt.Hyperband(
        hypermodel=lambda hp: build_model_with_hp(hp, normalizer),
        objective="val_auc",
        max_epochs=EPOCHS,
        factor=3,
        hyperband_iterations=2,
        project_name="wine_classification",
    )

    tuner.search_space_summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
        ),
    ]

    tuner.search(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\nBest hyperparameters found:")
    print(f"Learning Rate:    {best_hps.get('learning_rate'):.6f}")
    print(f"Hidden Units 1:   {best_hps.get('hidden_units_1')}")
    print(f"Dropout Rate:     {best_hps.get('dropout_rate'):.2f}")

    return best_hps, tuner


def main() -> int:
    # Load and preprocess data
    X_train, X_test, y_train, y_test, normalizer = load_wine_data()

    print("\n1. Tuning hyperparameters...")
    best_hps, tuner = tune_hyperparameters(X_train, y_train, normalizer)

    # Build and train best model
    best_model = create_model(
        normalizer=normalizer,
        learning_rate=best_hps.get("learning_rate"),
        hidden_units_1=best_hps.get("hidden_units_1"),
        dropout_rate=best_hps.get("dropout_rate"),
    )

    print("\n2. Training optimized model...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
        ),
    ]

    best_model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate best model
    print("\n3. Evaluating optimized model...")
    tuned_results = evaluate_model(best_model, X_test, y_test)

    # Compare results
    print(f"\n{'Metric':<20} {'Baseline':<15} {'Tuned':<15} {'Improvement'}")
    print("-" * 60)

    accuracy_improvement = tuned_results["accuracy"] - BASELINE["test_accuracy"]
    auc_improvement = tuned_results["auc"] - BASELINE["test_auc"]

    print(
        f"{'Test Accuracy':<20} "
        f"{BASELINE['test_accuracy']:>6.4f} ({BASELINE['test_accuracy'] * 100:>5.2f}%)  "
        f"{tuned_results['accuracy']:>6.4f} ({tuned_results['accuracy'] * 100:>5.2f}%)  "
        f"{accuracy_improvement:>+.4f} ({accuracy_improvement * 100:>+6.2f}%)"
    )
    print(
        f"{'Test AUC':<20} "
        f"{BASELINE['test_auc']:>6.4f}          "
        f"{tuned_results['auc']:>6.4f}          "
        f"{auc_improvement:>+.4f}"
    )

    best_model_path = "best_tuned_model.keras"
    best_model.save(best_model_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

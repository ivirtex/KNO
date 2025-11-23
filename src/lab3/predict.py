import argparse
import sys
from pathlib import Path

import numpy as np
import keras


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict wine category based on chemical analysis."
    )

    parser.add_argument("alcohol", type=float, help="Alcohol")
    parser.add_argument("malic_acid", type=float, help="Malic acid")
    parser.add_argument("ash", type=float, help="Ash")
    parser.add_argument("alcalinity_of_ash", type=float, help="Alcalinity of ash")
    parser.add_argument("magnesium", type=float, help="Magnesium")
    parser.add_argument("total_phenols", type=float, help="Total phenols")
    parser.add_argument("flavanoids", type=float, help="Flavanoids")
    parser.add_argument("nonflavanoid_phenols", type=float, help="Nonflavanoid phenols")
    parser.add_argument("proanthocyanins", type=float, help="Proanthocyanins")
    parser.add_argument("color_intensity", type=float, help="Color intensity")
    parser.add_argument("hue", type=float, help="Hue")
    parser.add_argument("od280_od315", type=float, help="OD280/OD315 of diluted wines")
    parser.add_argument("proline", type=float, help="Proline")

    args = parser.parse_args()

    # Ścieżka do najlepszego modelu
    model_path = Path(__file__).parent / "best_model.keras"

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        print("Please train the model first by running main.py", file=sys.stderr)
        return 1

    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)

    input_data = np.array(
        [
            [
                args.alcohol,
                args.malic_acid,
                args.ash,
                args.alcalinity_of_ash,
                args.magnesium,
                args.total_phenols,
                args.flavanoids,
                args.nonflavanoid_phenols,
                args.proanthocyanins,
                args.color_intensity,
                args.hue,
                args.od280_od315,
                args.proline,
            ]
        ]
    )

    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction[0]) + 1
    confidence = np.max(prediction[0]) * 100

    print(f"Predicted wine category: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nClass probabilities:")
    print(f"  Class 1: {prediction[0][0] * 100:.2f}%")
    print(f"  Class 2: {prediction[0][1] * 100:.2f}%")
    print(f"  Class 3: {prediction[0][2] * 100:.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())

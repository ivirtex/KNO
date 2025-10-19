import argparse
import sys

import tensorflow as tf


@tf.function
def solve_linear_system(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    return tf.linalg.solve(a, b)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a",
        type=float,
        nargs="+",
        required=True,
        help="Elements of the matrix A (row-major)",
    )
    parser.add_argument(
        "--b",
        type=float,
        nargs="+",
        required=True,
        help="Elements of the vector B",
    )
    args = parser.parse_args()

    n = len(args.b)

    if len(args.a) != n * n:
        print(
            f"Invalid dimensions: For {n} equations, matrix A needs {n * n} elements, but got {len(args.a)}."
        )
        return 1

    a = tf.constant(args.a, shape=(n, n))
    b = tf.constant(args.b, shape=(n, 1))

    det = tf.linalg.det(a)
    if tf.abs(det) < 1e-10:
        print(f"No unique solution exists (determinant = {det.numpy():.2e}).")

        return 1

    x = solve_linear_system(a, b)
    print(f"Solution: {x}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

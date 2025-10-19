import sys

import tensorflow as tf


def main() -> int:
    a = tf.constant([[3, 2], [1, 2]], shape=(2, 2), dtype=tf.float32)
    b = tf.constant([5, 4], shape=(2, 1), dtype=tf.float32)
    x = tf.linalg.solve(a, b)

    print(f"A: {a}")
    print(f"B: {b}")
    print(f"X: {x}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

from math import cos, sin
import argparse
import sys

import numpy as np
import tensorflow as tf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x", type=float, default=1.0, help="X coordinate of the point"
    )
    parser.add_argument(
        "--y", type=float, default=0.0, help="Y coordinate of the point"
    )
    parser.add_argument(
        "--angle", type=float, default=np.pi / 4, help="Rotation angle in radians"
    )
    args = parser.parse_args()

    point = tf.constant([args.x, args.y], shape=(2, 1), dtype=tf.float32)

    rotation_matrix = tf.constant(
        [[cos(args.angle), -sin(args.angle)], [sin(args.angle), cos(args.angle)]],
        dtype=tf.float32,
    )

    rotated_point = tf.matmul(rotation_matrix, point)

    print(f"Original point: {point}")
    print(f"Point after rotation: {rotated_point}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

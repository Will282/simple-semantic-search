import random
import struct

import numpy as np

from app.database import VectorSerializer


def test_serialize():
    vector = [1.1, 2.2, 3.3, 4.4]
    expected_bytes = struct.pack("4f", *vector)
    assert VectorSerializer.serialize(vector) == expected_bytes


def test_deserialize():
    byte_data = struct.pack("4f", 1.1, 2.2, 3.3, 4.4)
    expected_vector = [1.1, 2.2, 3.3, 4.4]
    np.testing.assert_allclose(VectorSerializer.deserialize(byte_data), expected_vector)


def test_round_trip():
    vector = [1.1, 2.2, 3.3, 4.4]
    byte_data = VectorSerializer.serialize(vector)
    result_vector = VectorSerializer.deserialize(byte_data)
    np.testing.assert_allclose(result_vector, vector)


def test_empty_list():
    vector = []
    expected_bytes = struct.pack("0f")
    assert VectorSerializer.serialize(vector) == expected_bytes
    assert VectorSerializer.deserialize(expected_bytes) == vector


def test_single_element():
    vector = [7.89]
    expected_bytes = struct.pack("1f", *vector)
    assert VectorSerializer.serialize(vector) == expected_bytes
    np.testing.assert_allclose(VectorSerializer.deserialize(expected_bytes), vector)


def test_large_list():
    vector = [random.random() for _ in range(1000)]
    byte_data = VectorSerializer.serialize(vector)
    result_vector = VectorSerializer.deserialize(byte_data)
    np.testing.assert_allclose(result_vector, vector)

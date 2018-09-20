import numpy as np

from mldissect.utils import multiply_row


def test_multiply_row():
    row = np.array([[1, 2, 3]])
    data = multiply_row(row, 3)
    expected = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert np.allclose(data, expected)

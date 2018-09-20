import numpy as np


def multiply_row(row, num_rows):
    return np.repeat(row, repeats=num_rows, axis=0)


def normalize_array(instance):
    return instance.reshape(1, -1)


def to_matrix(data):
    return data if not hasattr(data, 'values') else data.values

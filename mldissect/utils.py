import numpy as np
from collections import deque

def multiply_row(row, num_rows):
    return np.repeat(row, repeats=num_rows, axis=0)


def normalize_array(instance):
    return instance.reshape(1, -1)


def _get_means_from_yhats(important_yhats):
    data = [np.array(v).mean(axis=0) for v in important_yhats]
    return deque(data)

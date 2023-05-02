import numpy as np


def pad_vector(vector, fix_length):
    if len(vector) < fix_length:
        return np.pad(vector, (0, fix_length - len(vector) % fix_length), 'constant')
    else:
        return vector[:fix_length]


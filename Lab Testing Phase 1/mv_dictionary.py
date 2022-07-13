import numpy as np
from typing import List, Tuple
import scipy


class MVMapping:

    def __init__(self):
        self.keys = []
        self.values = []

    def __setitem__(self, key: List[Tuple[int, int, int, int]], value: Tuple[int, int]) -> None:
        """
        Add a key-value pair to the map.

        :param key: A vector field as a list of vectors.
        :param value: A tuple of the displacements in the x and y directions.
        """
        # Transform the vector field to a valid key.
        self.keys.append(MVMapping.vector_field_to_key(key))
        # Make sure value is a tuple that contains the displacements in the x and y directions.
        assert isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int) and isinstance(value[1], int)
        self.values.append(value)

    def __getitem__(self, item):
        """
        Get the displacements in the x and y directions from a vector field.

        :param item: A vector field as a list of vectors.
        :return: A tuple of the displacements in the x and y directions.
        """
        key = MVMapping.vector_field_to_key(item)
        best_match, best_cost = 0, float('inf')
        for i in range(len(self.keys)):
            cost = MVMapping.cost_function(self.keys[i], key)
            if cost < best_cost:
                best_match = i
                best_cost = cost
        return



    @staticmethod
    def vector_difference(new_vec: Tuple[int, int, int, int], key_vec: Tuple[int, int, int, int],
                          mode: int = 1) -> float:
        if key_vec == (0, 0):
            return 1 if new_vec == key_vec else 0
        new_vec = np.array([new_vec[2] - new_vec[0], new_vec[3] - new_vec[1]], dtype=np.float128)
        key_vec = np.array([key_vec[2] - key_vec[0], key_vec[3] - key_vec[1]], dtype=np.float128)
        if mode == 1:
            diff = np.abs(new_vec - key_vec).sum()
            key_vec = np.abs(key_vec).sum()
            return diff / key_vec
        elif mode == 2:
            diff = np.square(new_vec - key_vec).sum()
            key_vec = np.square(key_vec).sum()
            return np.sqrt(diff / key_vec)
        elif mode == 3:
            diff = np.abs(new_vec - key_vec).max()
            key_vec = np.abs(key_vec).max()
            if key_vec == 0:
                return 0
            return diff / key_vec

    @staticmethod
    def cost_function(key1, key2):
        diff = 0
        for i in range(len(key1)):
            diff += MVMapping.vector_difference(key2[i], key1[i])
        return diff

    @staticmethod
    def vector_field_to_key():
        pass

import numpy as np
from typing import List, Tuple
from scipy.spatial import KDTree


class MVMapping:
    """
    This class maps a list of vectors to a tuple representing its displacements in the x and y directions.
    The list of vectors are transformed to a kd-tree.
    """

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
        self.keys.append(KDTree(key, balanced_tree=True))
        # Make sure value is a tuple that contains the displacements in the x and y directions.
        assert isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int) and isinstance(value[1], int)
        self.values.append(value)

    def __getitem__(self, item: List[Tuple[int, int, int, int]]) -> Tuple[int, int]:
        """
        Get the displacements in the x and y directions from a vector field.

        :param item: A vector field as a list of vectors.
        :return: A tuple of the displacements in the x and y directions.
        """
        as_array = np.array(item)
        best_index = 0
        min_distance, min_index = self.keys[best_index].quary(as_array)[0]
        for i in range(len(self.keys)):
            distance = self.keys[i].quary(as_array)[0]
            if distance < min_distance:
                min_distance = distance
                best_index = i
        return self.values[best_index]


# HOW TO USE KDTREE:
# a = np.array([(1, 2), (2, 4), (1000, 1000)])
# b = KDTree(a)
# v = np.array([600, 600])
# print(b.query(v, k=3, p=23, workers=1))
# print(a[b.query(v)[1]])


import pickle
import numpy as np
from typing import List, Tuple, Any
from scipy.spatial import KDTree


class MVMapping:

    def __init__(self, save_file: str = None):
        self.keys = []
        self.values = []
        if save_file is not None:
            self.load_from(save_file)

    def __setitem__(self, key: List[Tuple[int, int, int, int]], value: Any):
        """
        Add a key-value pair to the map.

        :param key: A vector field as a list of vectors.
        :param value: A tuple of the displacements in the x and y directions.
        """
        vectors = np.array([v for v in key if v[0] != v[2] or v[1] != v[3]])
        if len(vectors) == 0:
            return
        self.keys.append(KDTree(vectors, balanced_tree=True))
        self.values.append(value)

    def __getitem__(self, item: List[Tuple[int, int, int, int]]) -> Any:
        """
        Get the displacements in the x and y directions from a vector field.

        :param item: A vector field as a list of vectors.
        :return: A tuple of the displacements in the x and y directions.
        """
        as_array = np.array([v for v in item if v[0] != v[2] or v[1] != v[3]])
        if len(as_array) == 0:
            return 0
        best_index = 0
        distances = self.keys[best_index].query(as_array)[0]
        min_distance = sum(distances) / len(distances)
        for i in range(1, len(self.keys)):
            distances = self.keys[i].query(as_array)[0]
            distance = sum(distances) / len(distances)
            if distance < min_distance:
                min_distance = distance
                best_index = i
        return self.values[best_index]

    def load_from(self, save_file: str, append: bool = False) -> None:
        """
        Load the keys and values of a MVMapping object from a file.

        :param save_file: The path to the save file.
        :param append: Load from a file without resetting.
        """
        if not append:
            if len(self.keys) != 0:
                self.keys = []
            if len(self.values) != 0:
                self.values = []
        with open(save_file, 'rb') as f:
            while True:
                try:
                    self.keys.append(pickle.load(f))
                    self.values.append(pickle.load(f))
                except EOFError:
                    break

    def save_to(self, save_file: str) -> None:
        """
        Save the keys and values of a MVMapping object to a file.

        :param save_file: The path to the save file.
        """
        with open(save_file, 'wb+') as f:
            for i in range(len(self.keys)):
                pickle.dump(self.keys[i], f)
                pickle.dump(self.values[i], f)

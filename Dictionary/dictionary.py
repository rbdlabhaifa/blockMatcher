import pickle
import numpy as np
from typing import List, Tuple, Any
from scipy.spatial import KDTree
from python_block_matching import BlockMatching


class MVMapping:

    def __init__(self, save_file: str = None):
        self.keys = []
        self.values = []
        if save_file is not None:
            self.load_from(save_file)

    def __setitem__(self, key: List[Tuple[int, int, int, int]], value: Tuple[int, int, int]):
        """
        Add a key-value pair to the map.

        :param key: A vector field as a list of vectors.
        :param value: A tuple of the displacements in the x and y directions.
        """
        vectors = MVMapping.remove_zeroes(key)
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
        as_array = MVMapping.remove_zeroes(item)
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

    def train_by_images(self, frame_after_motion, frame_before_motion, motion):
        """
        @param frame_after_motion: The frame after the motion happened (Type: numpy array of size (N, M, 3) RGB colors)
        @param frame_before_motion: The frame before the motion happened (Type: numpy array of size (N, M, 3) RGB colors)
        @param motion: The motion in to 6 DOF
        @return:
        """
        self[BlockMatching.get_motion_vectors(frame_after_motion, frame_before_motion)] = motion

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

    @staticmethod
    def remove_zeroes(vectors: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Removes vectors that have a magnitude of 0.

        :param vectors: A list of vectors.
        :return: A list of vectors without vectors of length 0.
        """
        new_vector_list = []
        for x1, y1, x2, y2 in vectors:
            if x1 != x2 or y1 != y2:
                new_vector_list.append((x1, y1, x2, y2))
        return np.array(new_vector_list)

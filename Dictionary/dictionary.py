import pickle
import numpy as np
from typing import List, Tuple, Any, Dict
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

    def compare(self, data_to_compare: Dict[float, List[Tuple[int, int, int, int]]], output_file: str):
        """
        Compares this object with another set of data.

        :param data_to_compare: A dictionary with rotation as keys and vectors as values.
        :param output_file: The file to save the results of the comparison to.
        """
        output = 'real rot, expected dict rot, actual dict rot, error from real rot, error from expected rot\n'
        total_expected_rot_error = 0
        total_real_rot_error = 0
        for real_rotation, motion_vectors in data_to_compare.items():
            dict_rot = round(self[motion_vectors], 3)
            real_rotation = round(real_rotation, 3)
            expected_dict_rot = round(min(self.values, key=lambda x: abs(x - real_rotation)), 3)
            error_from_real = round(100 * (abs(dict_rot - real_rotation) / real_rotation), 3)
            error_from_expected = round(100 * (abs(dict_rot - expected_dict_rot) / expected_dict_rot), 3)
            output += f'{real_rotation}, {expected_dict_rot}, {dict_rot}, {error_from_real}%, {error_from_expected}%\n'
            total_real_rot_error += error_from_real
            total_expected_rot_error += error_from_expected
        total_real_rot_error = round(total_real_rot_error / len(data_to_compare), 3)
        total_expected_rot_error = round(total_expected_rot_error / len(data_to_compare), 3)
        output += f'Average error from real rotations: {total_real_rot_error}%\n'
        output += f'Average error from expected rotations: {total_expected_rot_error}%\n'
        file = open(output_file, 'w')
        file.write(output)
        file.close()

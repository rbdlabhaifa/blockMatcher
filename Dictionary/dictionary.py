import os
import pickle
import cv2
import numpy as np
from typing import List, Tuple, Any, Dict
from scipy.spatial import KDTree
from block_matching import BlockMatching


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

    def load_from(self, save_file: str, append: bool = False):
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

    def save_to(self, save_file: str):
        """
        Save the keys and values of a MVMapping object to a file.

        :param save_file: The path to the save file.
        """
        with open(save_file, 'wb+') as f:
            for i in range(len(self.keys)):
                pickle.dump(self.keys[i], f)
                pickle.dump(self.values[i], f)

    def compare(self, data_to_compare: Dict[float, List[Tuple[int, int, int, int]]], output_file: str = None):
        """
        Compares this MVMapping object with a dictionary that contains rotation and vectors as items.

        :param data_to_compare: A dictionary with rotation as keys and motion vectors as values.
        :param output_file: The file to save the results of the comparison to.
        """
        # The output of the function. Will be saved to output_file or be printed if output_file is None.
        output = 'real rot, expected dict rot, actual dict rot, diff from real rot, diff from expected rot\n'
        # Keep the max difference.
        max_dict_to_real_diff = 0
        max_dict_to_expected_diff = 0
        # Keep the sum of difference.
        sum_dict_to_real_diff = 0
        sum_dict_to_expected_diff = 0
        for real_rot, motion_vectors in data_to_compare.items():
            # The rotation output from this MVMapping object.
            dict_rot = round(self[motion_vectors], 3)
            # The real rotation rounded to 3 decimal digits.
            real_rot = round(real_rot, 3)
            # The rotation that this object is expected to return.
            expected_dict_rot = round(min(self.values, key=lambda x: abs(x - real_rot)), 3)
            # The differences of the dict rotation from the real rotation.
            diff_from_real = round(abs(dict_rot - real_rot), 3)
            # The differences of the dict rotation from the expected dict rotation.
            diff_from_expected = round(abs(dict_rot - expected_dict_rot), 3)
            # Add a row to the output string.
            output += f'{real_rot}, {expected_dict_rot}, {dict_rot}, {diff_from_real}, {diff_from_expected}\n'
            # Add the differences to the sum.
            sum_dict_to_real_diff += diff_from_real
            sum_dict_to_expected_diff += diff_from_expected
            # Check for maximum difference.
            max_dict_to_real_diff = max(max_dict_to_real_diff, diff_from_real)
            max_dict_to_expected_diff = max(max_dict_to_expected_diff, diff_from_expected)
        # The average differences and max differences.
        output += f'Max difference from expected rotation: {max_dict_to_expected_diff}\n'
        output += f'Max difference from real rotation: {max_dict_to_real_diff}\n'
        output += f'Average diff from expected rotations: {round(sum_dict_to_expected_diff / len(data_to_compare), 3)}\n'
        output += f'Average diff from real rotations: {round(sum_dict_to_real_diff / len(data_to_compare), 3)}\n'
        if output_file is None:
            print(output)
        else:
            file = open(output_file, 'w')
            file.write(output)
            file.close()

    def create_from(self, path_to_data: str, angle_step: float = 0.1) -> Dict[float, List[Tuple[int, int, int, int]]]:
        """
        Adds data to this MVMapping object.

        :param path_to_data: The path to a directory that contains images or the path to a video created with ffmpeg.
        :param angle_step: The step in angle between the images in the data.
        :return: A dictionary that has angles as keys and lists of vectors as keys.
        """
        if path_to_data.endswith('.mp4') or path_to_data.endswith('.h264'):
            frames = BlockMatching.extract_motion_data(path_to_data)
        else:
            files = sorted(os.listdir(path_to_data), key=lambda x: int(x.replace('.png', '').replace('.jpg', '')))
            frames = BlockMatching.get_ffmpeg_motion_vectors_with_cache([f'{path_to_data}/{i}' for i in files])
        data = {}
        for i, mvs in enumerate(frames):
            if i % 2 == 1:
                continue
            angle = angle_step * (1 + (i // 2))
            self[mvs] = angle
            data[angle] = mvs
        return data

    @staticmethod
    def create_compare_data(path_to_data: str, angle_step: float = 0.1,
                            debug: bool = False) -> Dict[float, List[Tuple[int, int, int, int]]]:
        """
        Creates data to compare with MVMapping objects.

        :param path_to_data: The path to a directory that contains images or the path to a video created with ffmpeg.
        :param angle_step: The step in angle between the images in the data.
        :param debug: If True, vectors will be shown on the screen and the angle will be printed.
        :return: A dictionary that has angles as keys and lists of vectors as keys.
        """
        if path_to_data.endswith('.mp4') or path_to_data.endswith('.h264'):
            frames = BlockMatching.extract_motion_data(path_to_data)
        else:
            files = sorted(os.listdir(path_to_data), key=lambda x: int(x.replace('.png', '').replace('.jpg', '')))
            frames = BlockMatching.get_ffmpeg_motion_vectors_with_cache([f'{path_to_data}/{i}' for i in files])
        data = {}
        for i, mvs in frames:
            if i % 2 == 1:
                continue
            angle = angle_step * (1 + (i // 2))
            data[angle] = mvs
            if debug:
                print(f'i={i}, angle={angle}')
                base_frame = cv2.imread(frames[0])
                base_frame = BlockMatching.draw_motion_vectors(base_frame, mvs)
                cv2.imshow('debug', base_frame)
                cv2.waitKey()
        return data

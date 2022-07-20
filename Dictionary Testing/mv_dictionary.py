import math
import pickle
import numpy as np
from typing import List, Tuple
from scipy.spatial import KDTree
from python_block_matching import BlockMatching
from data_generator import DataGenerator


class MVMapping:

    def __init__(self, save_file: str = None):
        self.keys = []
        self.values = []
        if save_file is not None:
            self.load_from(save_file)

    def __setitem__(self, key: List[Tuple[int, int, int, int]], value: Tuple[int, int, int]) -> None:
        """
        Add a key-value pair to the map.

        :param key: A vector field as a list of vectors.
        :param value: A tuple of the displacements in the x and y directions.
        """
        self.keys.append(KDTree(MVMapping.remove_zeroes(key), balanced_tree=True))
        self.values.append(value)

    def __getitem__(self, item: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int]:
        """
        Get the displacements in the x and y directions from a vector field.

        :param item: A vector field as a list of vectors.
        :return: A tuple of the displacements in the x and y directions.
        """
        as_array = MVMapping.remove_zeroes(item)
        best_index = 0
        distances = self.keys[best_index].query(as_array)[0]
        min_distance = min((sum(distances) / len(distances)), 5)
        for i in range(1, len(self.keys)):
            distances = self.keys[i].query(as_array)[0]
            distance = min((sum(distances) / len(distances)), 5)
            if distance < min_distance:
                min_distance = distance
                best_index = i
        return self.values[best_index]

    def get_min_distances(self, vectors: List[Tuple[int, int, int, int]]) -> List[float]:
        """
        Get the distance of the vectors with the best tree.

        :param vectors: A list of vectors.
        :return: A list of distances.
        """
        as_array = MVMapping.remove_zeroes(vectors)
        distances = self.keys[0].query(as_array)[0]
        best_distances = distances
        min_distance = sum(distances) / len(distances)
        for i in range(1, len(self.keys)):
            distances = self.keys[i].query(as_array)[0]
            distance = sum(distances) / len(distances)
            if distance < min_distance:
                min_distance = distance
                best_distances = distances
        return best_distances

    def load_from(self, save_file: str) -> None:
        """
        Load the keys and values of a MVMapping object from a file.

        :param save_file: The path to the save file.
        """
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
        with open(save_file, 'wb') as f:
            for i in range(len(self.keys)):
                pickle.dump(self.keys[i], f)
                pickle.dump(self.values[i], f)

    def train_by_images(self, frame_after_motion, frame_before_motion, motion):
        """
        @param frame_after_motion: The frame after the motion happened (Type: numpy array of size (N, M, 3) RGB colors)
        @param frame_before_motion: The frame before the motion happened (Type: numpy array of size (N, M, 3) RGB colors)
        @param motion: The motion in to 6 DOF
        @return:
        """
        self[BlockMatching.get_motion_vectors(frame_after_motion, frame_before_motion)] = motion

    def train(self, image_path: str, start: int = 10, target: int = 110, step: int = 10) -> None:
        """
        Trains the dictionary for translation and rotation detection from an image.

        :param image_path: The path of the image to train the dictionary with.
        :param start: The start value of the translation.
        :param target: The end value of the translation.
        :param step: The step between each iteration.
        """
        image_dimensions = [360, 360]
        for flip in range(-1, 2, 2):
            for i in range(start, target, step):
                translation = (flip * i, 0, 0)
                f1, f2 = DataGenerator.generate_translation(image_dimensions, image_path, translation[:2])
                self[BlockMatching.get_motion_vectors(f2, f1)] = translation
                translation = (0, flip * i, 0)
                f1, f2 = DataGenerator.generate_translation(image_dimensions, image_path, translation[:2])
                self[BlockMatching.get_motion_vectors(f2, f1)] = translation
                translation = (flip * i, flip * i, 0)
                f1, f2 = DataGenerator.generate_translation(image_dimensions, image_path, translation[:2])
                self[BlockMatching.get_motion_vectors(f2, f1)] = translation
                translation = (-flip * i, flip * i, 0)
                f1, f2 = DataGenerator.generate_translation(image_dimensions, image_path, translation[:2])
                self[BlockMatching.get_motion_vectors(f2, f1)] = translation
        for i in range(0, 360):
            f1, f2 = DataGenerator.generate_rotation(image_dimensions, image_path, i)
            self[BlockMatching.get_motion_vectors(f2, f1)] = (0, 0, i)

    def try_dictionary(self, image: str) -> None:
        """
        A function to check the dictionary.

        :param image: The path to an image to test the dictionary with.
        """
        while True:
            mode = input('enter mode (t - translation, r - rotation, c - change image): ').lower()
            if mode == 'c':
                image = input('enter image path: ')
            elif mode == 't':
                translation = eval(input('enter translation (x, y): ').replace(' ', ','))
                f1, f2 = DataGenerator.generate_translation([360, 360], image, translation)
                print('dictionary found: ', self[BlockMatching.get_motion_vectors(f2, f1)])
            elif mode == 'r':
                rotation = int(input('enter rotation: '))
                f1, f2 = DataGenerator.generate_rotation([360, 360], image, rotation)
                print('dictionary found: ', self[BlockMatching.get_motion_vectors(f2, f1)])

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
        return np.array(vectors)

    @staticmethod
    def calculate_camera_x_rotation(vector: Tuple[int, int, int, int], resolution: Tuple[int, int], x_fov: int) -> float:
        """
        Calculates the rotation of the camera along the x-axis of the frame.

        :param vector: The motion vector.
        :param resolution: The resolution of the frame.
        :param x_fov: The angle of the camera's pinhole with the view plane on the x axis.
        :return: The rotation of the camera along the x-axis of the frame.
        """
        x_fov = np.deg2rad(x_fov)
        vector_length = abs(vector[2] - vector[0])
        cam_to_frame = (resolution[0] / 2) / np.tan(x_fov / 2)
        half_width = resolution[0] // 2
        distance_from_center = min(abs(half_width - vector[0]), abs(half_width - vector[2]))
        a = np.sqrt(cam_to_frame ** 2 + distance_from_center ** 2)
        b = np.sqrt(cam_to_frame ** 2 + (distance_from_center + vector_length) ** 2)
        return np.rad2deg(np.arcsin((cam_to_frame * vector_length) / (a * b)))

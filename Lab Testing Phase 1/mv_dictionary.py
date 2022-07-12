import math

import cv2

from python_block_matching import BlockMatching, BMFrame, BMVideo
import numpy as np
from typing import List


class MVDictUtils:
    @staticmethod
    def normalize_motion_vector_field(motion_vector_array):
        # mv_array = []
        # # Zero Vectors Filtering
        # for mv in motion_vector_array:
        #     if mv[0] != mv[2] or mv[1] != mv[3]:
        #         mv_array.append(mv)
        # min_x, min_y = mv_array[0][:2]
        #
        # out = []
        # for mv in mv_array:
        #     out.append((mv[0] - min_x, mv[1] - min_y, mv[2] - min_x, mv[3] - min_y))
        # return out
        return motion_vector_array


class MVDict(dict):

    @staticmethod
    def vector_diff(new_vec, key_vec, norm=1):
        if key_vec == (0, 0):
            return 1 if new_vec == key_vec else 0
        new_vec = np.array([new_vec[2] - new_vec[0], new_vec[3] - new_vec[1]], dtype=np.float128)
        key_vec = np.array([key_vec[2] - key_vec[0], key_vec[3] - key_vec[1]], dtype=np.float128)
        if norm == 1:
            diff = np.abs(new_vec - key_vec).sum()
            key_vec = np.abs(key_vec).sum()
            return diff / key_vec
        elif norm == 2:
            diff = np.square(new_vec - key_vec).sum()
            key_vec = np.square(key_vec).sum()
            return np.sqrt(diff / key_vec)
        else:
            diff = np.abs(new_vec - key_vec).max()
            key_vec = np.abs(key_vec).max()
            if key_vec == 0:
                return 0
            return diff / key_vec

    @staticmethod
    def cost_function(key1, key2):
        diff = 0
        for i in range(len(key1)):
            diff += MVDict.vector_diff(key2[i], key1[i])
        return diff

    def __setitem__(self, key, value: List[int]):
        """
        @param key: Array Of Motion Vectors as produced by BlockMatching.get_motion_vectors()
        @param value: (x_displacement, y_displacement)
        """
        assert len(value) == 2
        key = MVDictUtils.normalize_motion_vector_field(key)

        super(MVDict, self).__setitem__(tuple(key), value)

    def __getitem__(self, item):
        item = MVDictUtils.normalize_motion_vector_field(item)
        min_cost = float("inf")
        min_key = None

        for key in self.keys():
            if min_cost > self.cost_function(key, item):
                min_cost = self.cost_function(key, item)
                min_key = key

        return super(MVDict, self).__getitem__(tuple(min_key))


if __name__ == '__main__':
    from python_block_matching import *
    from data_generator import DataGenerator

    frame = BMFrame(cv2.imread("Benchmark_Pictures/Black.png"))
    mv_dict = MVDict()
    # Generate mv_dict
    for i in range(1, 10):
        translation = [i * 10, 0]
        f1, f2 = DataGenerator.generate_movement([160 + 16 * 6, 160 + 16 * 6], "Benchmark_Pictures/Black.png",
                                                 tuple(translation))
        mvs = [*BlockMatching.get_motion_vectors(f2, f1)]
        mvs = list(filter(lambda x: x[:2] != x[2:], mvs))
        mv_dict[tuple(mvs)] = translation
        translation = [0, i * 10]
        f1, f2 = DataGenerator.generate_movement([160 + 16 * 6, 160 + 16 * 6], "Benchmark_Pictures/Black.png",
                                                 tuple(translation))
        mvs = [*BlockMatching.get_motion_vectors(f2, f1)]
        mv_dict[tuple(mvs)] = translation

    translation = [40, 0]
    f1, f2 = DataGenerator.generate_movement([160 + 16 * 6, 160 + 16 * 6], "Benchmark_Pictures/Black.png",
                                             tuple(translation))
    mvs = [*BlockMatching.get_motion_vectors(f2, f1)]
    print(mv_dict[tuple(mvs)]
          )

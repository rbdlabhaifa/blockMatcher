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
    def cost_function(key1, key2):
        s_errors = []
        a_errors = []
        for i in range(len(key1)):
            x1, y1, x2, y2 = key1[i]
            size1 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if x2 - x1 == 0:
                angle1 = math.pi / 2
            else:
                angle1 = np.arctan(np.abs(y2 - y1) / np.abs(x2 - x1))
            x3, y3, x4, y4 = key2[i]
            size2 = np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2)
            if x4 - x3 == 0:
                angle2 = math.pi / 2
            else:
                angle2 = np.arctan(np.abs(y4 - y3) / np.abs(x4 - x3))

            size_diff = np.abs(size2 - size1)
            angle_diff = np.abs(angle2 - angle1)
            s_errors.append(size_diff)
            a_errors.append(angle_diff)
        return (sum(s_errors) / len(s_errors) + sum(a_errors) / len(a_errors)) / 2

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

    translation = [20, 0]
    f1, f2 = DataGenerator.generate_movement([160 + 16 * 6, 160 + 16 * 6], "Benchmark_Pictures/Black.png",
                                             tuple(translation))
    mvs = [*BlockMatching.get_motion_vectors(f2, f1)]
    print(mv_dict[tuple(mvs)]
          )

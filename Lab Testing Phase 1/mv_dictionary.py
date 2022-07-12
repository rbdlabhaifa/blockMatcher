from python_block_matching import BlockMatching, BMFrame, BMVideo
import numpy as np
from typing import List


class MVDictUtils:
    @staticmethod
    def normalize_motion_vector_field(motion_vector_array):
        mv_array = []
        # Zero Vectors Filtering
        for mv in motion_vector_array:
            if not np.array_equal(mv[0], mv[1]):
                mv_array.append(mv)

        min_x, min_y = mv_array[0]

        out = []
        for mv in mv_array:
            out.append((mv[0] - min_x, mv[1] - min_y, mv[2] - min_x, mv[3] - min_y))
        return out


class MVDict(dict):
    @staticmethod
    def cost_function(self):
        pass

    def __setitem__(self, key, value: List[int]):
        """
        @param key: Array Of Motion Vectors as produced by BlockMatching.get_motion_vectors()
        @param value: (x_displacement, y_displacement)
        """
        assert len(value) == 2
        key = MVDictUtils(key)
        super(MVDict, self).__setitem__(key, value)

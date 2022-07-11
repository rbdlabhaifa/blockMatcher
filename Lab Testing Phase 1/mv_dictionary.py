from python_block_matching import BlockMatching, BMFrame, BMVideo
import numpy as np
from typing import List, Tuple


class MVDictUtils:

    @staticmethod
    def normalize_motion_vector_field(motion_vectors: List[Tuple[int, int, int, int]], origin: Tuple[int, int] = (0, 0),
                                      return_full_vector: bool = True) -> List[Tuple[int, int]]:
        """
        Normalizes a list of motion vectors.

        :param motion_vectors: A list of the start and end points of the motion vectors.
        :param origin: The point to normalize the vector to.
        :param return_full_vector: True if you want 2 points for each tuple or False for just the end point.
        :return: A list of normalized vectors.
        """
        normalized_vectors = []
        if return_full_vector:
            for x1, y1, x2, y2 in motion_vectors:
                # Ignore vectors with a magnitude of 0.
                if x1 != x2 or y1 != y2:
                    normalized_vectors.append((origin[0], origin[1], x2 - x1 + origin[0], y2 - y1 + origin[1]))
        else:
            for x1, y1, x2, y2 in motion_vectors:
                # Ignore vectors with a magnitude of 0.
                if x1 != x2 or y1 != y2:
                    normalized_vectors.append((x2 - x1 + origin[0], y2 - y1 + origin[1]))
        return normalized_vectors


class MVDict(dict):

    @staticmethod
    def cost_function(key):
        return key

    def __setitem__(self, key, value: List[int]):
        """
        @param key: Array Of Motion Vectors as produced by BlockMatching.get_motion_vectors()
        @param value: (x_displacement, y_displacement)
        """
        assert len(value) == 2
        key = MVDict.cost_function(MVDictUtils.normalize_motion_vector_field(key, (0, 0), False))
        super(MVDict, self).__setitem__(key, value)


if __name__ == '__main__':
    from python_block_matching import *
    from data_generator import DataGenerator

    a = MVDict()
    f = BMVideo([f'Benchmark_Pictures/Image{i}.png' for i in range(15)])
    for i in range(f.get_frame_count()):
        frame = f[i]
        f1, f2 = DataGenerator.generate_movement([1000, 1000], f.frames[i], (20, 0))
        mvs = [*BlockMatching.get_motion_vectors(f2, f1)]
        # mvs = MVDictUtils.normalize_motion_vector_field(mvs, (500, 500))
        f1, f2 = BMFrame(f1), BMFrame(f2)
        f1.draw_macro_block(*BlockMatching.get_macro_blocks(f1.base_image), color=(255, 0, 0), thickness=1)
        f1.draw_motion_vector(*mvs, color=(0, 255, 0), thickness=1)
        f1.show()
        f2.show()

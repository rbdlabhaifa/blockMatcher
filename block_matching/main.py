from typing import Tuple, Union, List, Dict
import numpy as np
import cv2
import subprocess

from .block_matching import SEARCH_FUNCTIONS
from .block_partitioning import PARTITIONING_FUNCTION


class BlockMatching:

    @staticmethod
    def extract_motion_data(*frames: np.ndarray) -> None:
        subprocess.run('')


    @staticmethod
    def get_motion_vectors(current_frame: np.ndarray, reference_frame: np.ndarray,
                           search_function: str = 'DS', cost_function: str = 'SAD', step_size: int = 0,
                           current_frame_mb: List[Tuple[int, int, int, int]] = None) -> List[Tuple[int, int, int, int]]:
        """
        Generates motion vectors from the reference frame to the current frame.

        :param current_frame: The current frame as a numpy array.
        :param reference_frame: The reference frame as a numpy array.
        :param search_function: The function to search for the best macro-blocks with.
        :param cost_function: The cost function to use for the search.
        :param step_size: The step size to use in the search (0 to use the default of the search function).
        :param current_frame_mb: The macro-blocks int the current frame to run the search on.
        :return: The start and end points of the vectors.
        """
        if current_frame_mb is None:
            current_frame_mb = BlockMatching.get_macro_blocks(current_frame)
        motion_vectors = []
        search_function = SEARCH_FUNCTIONS[search_function]
        for x, y, w, h in current_frame_mb:
            sx, sy = search_function(current_frame, reference_frame, x, y, w, h, cost_function, step_size)
            hw, hh = w // 2, h // 2
            motion_vectors.append((sx + hw, sy + hh, x + hw, y + hh))
        return motion_vectors

    @staticmethod
    def get_macro_blocks(frame: np.ndarray, block_width: int = 16, block_height: int = 16,
                         partition_function: str = 'FIXED',
                         cost_function: str = 'SAD') -> List[Tuple[int, int, int, int]]:
        """
        Generate macro-blocks in a frame.

        :param frame: The frame generate the macro-blocks for.
        :param block_width: The max width of the macro-blocks.
        :param block_height: The max height of the macro-blocks.
        :param partition_function: The function to partition the frame with.
        :param cost_function: The cost function to use for the partitioning function.
        :return: A list that contains the top-left point of the macro-blocks and their width and height.
        """
        macro_blocks = []
        partition_function = PARTITIONING_FUNCTION[partition_function]
        for x, y, w, h in partition_function(frame, block_width, block_height, cost_function):
            macro_blocks.append((x, y, w, h))
        return macro_blocks

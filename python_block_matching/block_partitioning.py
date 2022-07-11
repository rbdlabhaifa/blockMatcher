from typing import Tuple
import numpy as np

from .cost_functions import COST_FUNCTIONS


def fixed_size_macro_blocks(frame: np.ndarray, block_width: int,
                            block_height: int, *args) -> Tuple[int, int, np.ndarray]:
    """
    Get the macro-blocks in a frame and their top-left points.

    :param frame: The frame as a numpy array.
    :param block_width: The width of the macro-block.
    :param block_height: The height of the macro-block.
    :return: The top-left points and of all macro-blocks in a frame and the macro-blocks.
    """
    frame_height, frame_width = frame.shape[:2]
    for y in range(0, block_height * (frame_height // block_height), block_height):
        for x in range(0, block_width * (frame_width // block_width), block_width):
            yield x, y, frame[y:y + block_height, x:x + block_width]


def variable_size_macro_blocks(frame: np.ndarray, block_width: int,
                               block_height: int, cost_function: str = 'SAD') -> Tuple[int, int, np.ndarray]:
    """
    Get the macro-blocks in a frame and their top-left points.
    The size of the blocks is one of the following: w * h, (w / 2) X h, w * (h / 2), (w / 2) * (h / 2).

    :param frame: The frame as a numpy array.
    :param block_width: The max width of the macro-block.
    :param block_height: The max height of the macro-block.
    :param cost_function: The cost function to compare macro-blocks partitions with.
    :return: The top-left points and of all macro-blocks in a frame and the macro-blocks.
    """
    cost_function = COST_FUNCTIONS[cost_function]
    hw, hh = block_width // 2, block_height // 2
    for x, y, mb in fixed_size_macro_blocks(frame, block_width, block_height):
        v1, v2 = mb[:, :hw], mb[:, hw:]
        h1, h2 = mb[:hh, :], mb[hh:, :]
        vertical_cost = cost_function(v1, v2)
        horizontal_cost = cost_function(h1, h2)
        threshold = 10000
        if vertical_cost > threshold and horizontal_cost > threshold:
            yield x, y, mb[:hh, :hw]
            yield x + hw, y, mb[:hh, hw:]
            yield x, y + hh, mb[hh:, :hw]
            yield x + hw, y + hh, mb[hh:, hw:]
        elif vertical_cost > threshold:
            yield x, y, v1
            yield x + hw, y, v2
        elif horizontal_cost > threshold:
            yield x, y, h1
            yield x, y + hh, h2
        else:
            yield x, y, mb


PARTITIONING_FUNCTION = {
    'FIXED':    fixed_size_macro_blocks,
    'VARIABLE': variable_size_macro_blocks
}

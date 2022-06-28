import numpy as np


def slice_macro_block(x: int, y: int, frame: np.ndarray, size: int) -> np.ndarray:
    """
    Returns a macro-block from a frame.

    :param x: X-coordinate of the top-left corner of the macro-block.
    :param y: Y-coordinate of the top-left corner of the macro-block.
    :param frame: The frame to slice the macro-block from (should be a numpy array containing RGB triples).
    :param size: The width and height of the macro-block.
    :return: A numpy array as a slice of the frame.
    """
    x, y = max(0, x), max(0, y)
    x, y = min(x, frame.shape[1] - size), min(y, frame.shape[0] - size)
    return frame[y:y + size, x:x + size]


def get_all_center_points(frame: np.ndarray, macro_block_size: int) -> tuple:
    """
    Returns a numpy array that contains the center points of all macro-blocks in the frame.

    :param frame: The frame as a numpy array that contains RGB triples.
    :param macro_block_size: The size (width / height) of the macro-blocks.
    :return: A numpy array of all the center points of the macro-blocks.
    """
    height, width = frame.shape[0], frame.shape[1]
    half_size = macro_block_size // 2
    for y in range(0, height, macro_block_size):
        for x in range(0, width, macro_block_size):
            yield x + half_size, y + half_size

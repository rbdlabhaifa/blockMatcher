import numpy as np


def get_macro_blocks(frame: np.ndarray, size: int) -> tuple:
    """
    Returns a numpy array that contains the center points of all macro-blocks in the frame.

    :param frame: The frame as a numpy array that contains RGB triples.
    :param size: The size (width / height) of the macro-blocks.
    :return: A numpy array of all the center points of the macro-blocks.
    """
    for y in range(0, size * (frame.shape[0] // size), size):
        for x in range(0, size * (frame.shape[1] // size), size):
            yield (x, y), frame[y:y + size, x:x + size]


def slice_macro_block(frame: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    """
    Slice and return a macro-block from a frame.
     The macro-block's size is guaranteed, but there is no guarantee that (x, y) will be the top-left corner.

    :param frame: The frame to slice a block from.
    :param x: The X-coordinate of the top-left corner of the macro-block.
    :param y: The Y-coordinate of the top-left corner of the macro-block.
    :param size: The size of the macro-block.
    :return: The macro-block.
    """
    x, y = max(x, 0), max(y, 0)
    x, y = min(x, frame.shape[1] - size), min(y, frame.shape[0] - size)
    return frame[y:y + size, x:x + size]

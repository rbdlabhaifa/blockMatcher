import numpy as np


def slice_macro_block(frame: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Slice and return a macro-block from a frame.

    The macro-block's size is guaranteed, but there is no guarantee that (x, y) will be the top-left corner.

    :param frame: The frame to slice a block from.
    :param x: The X-coordinate of the top-left corner of the macro-block.
    :param y: The Y-coordinate of the top-left corner of the macro-block.
    :param width: The width of the macro-block.
    :param height: The height of the macro-block.
    :return: The macro-block.
    """
    x, y = max(x, 0), max(y, 0)
    x, y = min(x, frame.shape[1] - width), min(y, frame.shape[0] - height)
    return frame[y:y + height, x:x + width]

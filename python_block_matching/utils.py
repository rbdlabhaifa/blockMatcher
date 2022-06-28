import numpy as np


def get_macro_block(x: int, y: int, frame: np.ndarray, macro_block_size: int) -> np.ndarray:
    """

    :param x: X-coordinate of the top-left corner of the desired macro-block.
    :param y: Y-coordinate of the top-left corner of the desired macro-block.
    :param frame: The frame to slice the macro-block from (should be a numpy ndarray containing RGB triples).
    :param macro_block_size: The width and height of the macro-block.
    :return: A numpy array as a slice of the frame.
    """
    if macro_block_size + y <= frame.shape[0] and macro_block_size + x <= frame.shape[1]:
        return frame[y:macro_block_size + y, x:macro_block_size + x]
    return frame[frame.shape[0] - macro_block_size:, frame.shape[1] - macro_block_size:]

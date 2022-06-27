import numpy as np
from Utils import get_macro_block
from typing import Callable
from cost_functions import mad


def three_step_search(current_frame: np.ndarray, reference_frame: np.ndarray, block_size: int,
                      cost_function: Callable = mad) -> tuple:
    """
    Three-Step Search Algorithm.

    :param current_frame: A Numpy array containing an RGB triple for every pixel in the current frame.
    :param reference_frame: A Numpy array containing an RGB triple for every pixel in the reference frame.
    :param block_size: An int representing the size (width / height) of a macro-block.
    :return: A tuple containing the x-y coordinates of the center pixel of the best matching macro-block.
    :param cost_function: The cost function to use (MAD by default).
    """
    s = 4  # step size = 4
    x, y = current_frame.shape[0] // 2, current_frame.shape[1] // 2  # start searching in the center
    best_block, min_score = None, float('inf')  # find the block with the minimum difference
    while s > 0:
        p1 = (x, y)
        p2 = (x + s, y)
        p3 = (x, y + s)
        p4 = (x + s, y + s)
        p5 = (x - s, y)
        p6 = (x, y - step)
        p7 = (x - s, y - s)
        p8 = (x + s, y - s)
        p9 = (x - s, y + s)
        for p in (p1, p2, p3, p4, p5, p6, p7, p8, p9):
            block_around_p = get_macro_block(p[0] - block_size // 2, p[1] - block_size // 2, )
            cost = cost_function(p, )
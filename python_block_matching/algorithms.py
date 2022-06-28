from typing import Callable

import numpy as np

from cost_functions import mad
from utils import get_macro_block


def three_step_search(current_frame_block: np.ndarray, reference_frame_search_area: np.ndarray, block_size: int,
                      cost_function: Callable = mad) -> tuple:
    """
    Three-Step Search Algorithm.

    :param current_frame_block: A macro-block in the current frame.
    :param reference_frame_search_area: The search area in the reference frame, should be 7x7.
    :param block_size: An int representing the size (width / height) of a macro-block.
    :param cost_function: The cost function to use (MAD by default).
    :return: A tuple containing the x-y coordinates of the center pixel of the best matching macro-block.
    """
    step = 4
    x = current_frame_block.shape[0] // 2
    y = current_frame_block.shape[1] // 2
    best_macro_block = None
    min_cost = float('inf')
    while step >= 1:
        p1 = (x, y)
        p2 = (x + step, y)
        p3 = (x, y + step)
        p4 = (x + step, y + step)
        p5 = (x - step, y)
        p6 = (x, y - step)
        p7 = (x - step, y - step)
        p8 = (x + step, y - step)
        p9 = (x - step, y + step)
        for p in (p1, p2, p3, p4, p5, p6, p7, p8, p9):
            p_macro_block = get_macro_block(p[0], p[1], reference_frame_search_area, block_size)
            cost = cost_function(current_frame_block, p_macro_block, block_size)
            if cost < min_cost:
                min_cost = cost
                best_macro_block = p
        step //= 2
    return best_macro_block


def two_dimensional_log_search():
    pass

import numpy as np

from .cost_functions import *
from .utils import *


def three_step_search(frame1: np.ndarray, frame2: np.ndarray, x: int, y: int, block_size: int,
                      cost_function: str = 'MAD') -> tuple:
    """
    Three-Step Search Algorithm.

    :param frame1: The reference frame as a numpy array containing RGB triples.
    :param frame2: The current frame as a numpy array containing RGB triples.
    :param x: The X-coordinate of the top-left of a macro-block in the current frame.
    :param y: The Y-coordinate of the top-left of a macro-block in the current frame.
    :param block_size: The size (width and height) of the macro-block.
    :param cost_function: The cost function to use, can only be either 'MAD' or 'MSE'.
    :return: The (x, y) of the center of the best matching macro-block in the reference frame.
    """
    step = 4
    search_size = 7
    frame2_block = slice_macro_block(frame2, x, y, block_size)
    search_area = slice_macro_block(frame1, x - search_size, y - search_size, block_size + search_size * 2)
    cx, cy = search_area.shape[0] // 2, search_area.shape[1] // 2
    min_cost, best_block = float('inf'), (cx, cy)
    if cost_function == 'MAD':
        cost_function = mad
    elif cost_function == 'MSE':
        cost_function = mse
    else:
        raise ValueError(f'Cost function can only be MAD or MSE, not {cost_function}.')
    while step >= 1:
        p1 = (cx, cy)
        p2 = (cx + step, cy)
        p3 = (cx, cy + step)
        p4 = (cx + step, cy + step)
        p5 = (cx - step, cy)
        p6 = (cx, cy - step)
        p7 = (cx - step, cy - step)
        p8 = (cx + step, cy - step)
        p9 = (cx - step, cy + step)
        for p in (p1, p2, p3, p4, p5, p6, p7, p8, p9):
            frame1_block = slice_macro_block(frame1,
                                             p[0] - block_size // 2,
                                             p[1] - block_size // 2,
                                             block_size)
            cost = cost_function(frame2_block, frame1_block)
            if cost < min_cost:
                min_cost = cost
                best_block = p
            step //= 2
    return x + best_block[0] - search_size, y + best_block[1] - search_size


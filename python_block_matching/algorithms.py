import numpy as np

from .cost_functions import *
from .utils import *


def three_step_search(current_frame: np.ndarray, reference_frame: np.ndarray, x: int, y: int, block_size: int,
                      cost_function: str = 'MAD') -> tuple:
    """
    Three-Step Search Algorithm.

    :param current_frame: The current frame as a numpy array containing RGB triples.
    :param reference_frame: The reference frame as a numpy array containing RGB triples.
    :param x: The X-coordinate of the center of a macro-block in the current frame.
    :param y: The Y-coordinate of the center of a macro-block in the current frame.
    :param block_size: The size (width and height) of the macro-block.
    :param cost_function: The cost function to use, can only be either 'MAD' or 'MSE'.
    :return: The (x, y) of the center of the best matching macro-block in the reference frame.
    """
    # Constants that should not be changed.
    step = 4
    search_size = 7
    # Macro-block in the current frame and search-area in the reference frame.
    top_left_x, top_left_y = x - block_size // 2, y - block_size // 2
    current_frame_block = slice_macro_block(top_left_x, top_left_y, current_frame, block_size)
    reference_frame_search_area = slice_macro_block(top_left_x - search_size // 2, top_left_y - search_size // 2,
                                                    reference_frame, search_size + block_size)
    # Choose a cost function to use.
    if cost_function == 'MAD':
        cost_function = mad
    elif cost_function == 'MSE':
        cost_function = mse
    else:
        raise ValueError('cost_function has to be \'MAD\' or \'MSE\'.')
    # The macro-block with the minimum cost value is the best match.
    best_macro_block, min_cost = (x, y), float('inf')
    while step >= 1:
        # Calculate all the points around (x, y).
        p1 = (x, y)
        p2 = (x + step, y)
        p3 = (x, y + step)
        p4 = (x + step, y + step)
        p5 = (x - step, y)
        p6 = (x, y - step)
        p7 = (x - step, y - step)
        p8 = (x + step, y - step)
        p9 = (x - step, y + step)
        # Calculate the cost for each point and choose the best macro-block.
        for p in (p1, p2, p3, p4, p5, p6, p7, p8, p9):
            p_macro_block = slice_macro_block(p[0], p[1], reference_frame_search_area, block_size)
            # FIXME: for some reason this returns the same score over and over again...
            cost = cost_function(current_frame_block, p_macro_block, block_size)
            if cost < min_cost:
                min_cost = cost
                best_macro_block = p
        # Cut the step distance by 2.
        step //= 2
    return best_macro_block

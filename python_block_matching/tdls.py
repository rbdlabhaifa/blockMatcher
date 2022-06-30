from .cost_functions import *
from .utils import *


def diamond_search(current_frame: np.ndarray, reference_frame: np.ndarray, x: int, y: int, block_size: int,
                   cost_function: str = 'MAD', step_size: int = 2) -> Tuple[int, int]:
    """
    Diamond Search Algorithm for Block-Matching.

    :param current_frame: The current frame gray-scaled and represented as a numpy array.
    :param reference_frame: The reference frame gray-scaled and represented as a numpy array.
    :param x: The X-coordinate of the top-left point of a macro-block in the current frame to perform the search on.
    :param y: The Y-coordinate of the top-left point of a macro-block in the current frame to perform the search on.
    :param block_size: The size of the macro-block.
    :param cost_function: The cost function to use. Can be either 'MAD' or 'MSE'.
    :param step_size: distance of searched points from the search origin. (SHOULD NOT BE CHANGED!)
    :return: The center point of the best matching macro-block in the reference frame.
    """
    # Set the cost function to be 'mad' or 'mse'.
    if cost_function == 'MAD':
        cost_function = mad
    elif cost_function == 'MSE':
        cost_function = mse
    else:
        raise ValueError(f'cost_function can only be \'MAD\' or \'MSE\', not {cost_function}.')
    # Get the macro-block from the current frame.
    macro_block = slice_macro_block(current_frame, x, y, block_size)
    # Set the starting point as the center of the macro-block.
    half_block_size = block_size // 2
    cx, cy = x + half_block_size, y + half_block_size
    # Keep the cost of (cx, cy), and update it and (cx, cy) if a block with smaller cost is found.
    min_cost = float('inf')

    # Large Diamond Search Pattern
    while step_size > 1:
        # Calculate all points in all 8 direction + the starting point.
        p1 = (cx, cy)
        p2 = (cx + step_size, cy)
        p3 = (cx, cy + step_size)
        p4 = (cx, cy - step_size)
        p5 = (cx - step_size, cy)
        p6 = (cx + step_size // 2, cy + step_size // 2)
        p7 = (cx - step_size // 2, cy - step_size // 2)
        p8 = (cx + step_size // 2, cy - step_size // 2)
        p9 = (cx - step_size // 2, cy + step_size // 2)
        for p in (p1, p2, p3, p4, p5, p6, p7, p8, p9):
            # Get the macro-block of the point from reference frame.
            ref_macro_block = slice_macro_block(reference_frame, p[0] - half_block_size, p[1] - half_block_size,
                                                block_size)
            # Calculate the cost between the blocks.
            cost = cost_function(macro_block, ref_macro_block)
            # Compare the cost to the minimum and update (cx, cy) and min_cost if a smaller cost was calculated.
            if cost < min_cost:
                min_cost = cost
                cx, cy = p
        # Cut the steps by 2.
        if (cx, cy) == p1:
            step_size //= 2

    # Small Diamond Search Pattern
    p1 = (cx, cy)
    p2 = (cx + step_size, cy)
    p3 = (cx - step_size, cy)
    p4 = (cx, cy - step_size)
    p5 = (cx, cy + step_size)
    for p in (p1, p2, p3, p4, p5):
        # Get the macro-block of the point from reference frame.
        ref_macro_block = slice_macro_block(reference_frame, p[0] - half_block_size, p[1] - half_block_size,
                                            block_size)
        # Calculate the cost between the blocks.
        cost = cost_function(macro_block, ref_macro_block)
        # Compare the cost to the minimum and update (cx, cy) and min_cost if a smaller cost was calculated.
        if cost < min_cost:
            min_cost = cost
            cx, cy = p
    return cx, cy

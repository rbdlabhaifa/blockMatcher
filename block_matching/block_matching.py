from typing import Tuple
import numpy as np

from .cost_functions import COST_FUNCTIONS
from .utils import slice_macro_block


def three_step_search(current_frame: np.ndarray,
                      reference_frame: np.ndarray,
                      x: int,
                      y: int,
                      width: int,
                      height: int,
                      cost_function: str = 'MAD',
                      step_size: int = 4) -> Tuple[int, int]:
    """
    Three-Step Search Algorithm for Block-Matching.

    :param current_frame: The current frame as a numpy array.
    :param reference_frame: The reference frame as a numpy array.
    :param x: The X-coordinate of the top-left point of a macro-block in the current frame to perform the search on.
    :param y: The Y-coordinate of the top-left point of a macro-block in the current frame to perform the search on.
    :param width: Width of the macro block.
    :param height: Height of the macro block.
    :param cost_function: The cost function to use.
    :param step_size: The distance of points to search from the origin search point. (SHOULD NOT BE CHANGED!)
    :return: The top-left point best macro-block in reference frame.
    """
    # Set step.
    step_size = step_size if step_size != 0 else 2
    # Set the cost function.
    cost_function = COST_FUNCTIONS[cost_function]
    # Get the macro-block from the current frame.
    macro_block = slice_macro_block(current_frame, x, y, width, height)
    # Set the starting point as the center of the macro-block.
    half_block_size = (width // 2, height // 2)
    cx, cy = x + half_block_size[0], y + half_block_size[1]
    # Keep the cost of (cx, cy), and update it and (cx, cy) if a block with smaller cost is found.
    min_cost = float('inf')
    while step_size >= 1:
        # Calculate all points in all 8 direction + the starting point.
        p1 = (cx, cy)
        p2 = (cx + step_size, cy)
        p3 = (cx, cy + step_size)
        p4 = (cx + step_size, cy + step_size)
        p5 = (cx - step_size, cy)
        p6 = (cx, cy - step_size)
        p7 = (cx - step_size, cy - step_size)
        p8 = (cx + step_size, cy - step_size)
        p9 = (cx - step_size, cy + step_size)
        for p in (p1, p2, p3, p4, p5, p6, p7, p8, p9):
            # Get the macro-block of the point from reference frame.
            ref_macro_block = slice_macro_block(reference_frame, p[0] - half_block_size[0], p[1] - half_block_size[1],
                                                width, height)
            # Calculate the cost between the blocks.
            cost = cost_function(macro_block, ref_macro_block)
            # Compare the cost to the minimum and update (cx, cy) and min_cost if a smaller cost was calculated.
            if cost < min_cost:
                min_cost = cost
                cx, cy = p
        # Cut the steps by 2.
        step_size //= 2
    return cx - half_block_size[0], cy - half_block_size[1]


def two_dimensional_logarithmic_search(current_frame: np.ndarray,
                                       reference_frame: np.ndarray,
                                       x: int,
                                       y: int,
                                       width: int,
                                       height: int,
                                       cost_function: str = 'MAD',
                                       step_size: int = 4) -> Tuple[int, int]:
    """
    Two-Dimensional Logarithmic Search for Block-Matching.

    :param current_frame: The current frame as a numpy array.
    :param reference_frame: The reference frame as a numpy array.
    :param x: The X-coordinate of the top-left point of a macro-block in the current frame to perform the search on.
    :param y: The Y-coordinate of the top-left point of a macro-block in the current frame to perform the search on.
    :param width: Width of the macro block.
    :param height: Height of the macro block.
    :param cost_function: The cost function to use.
    :param step_size: The distance of points to search from the origin search point.
    :return: The top-left point of the best macro-block in reference frame.
    """
    # Set step.
    step_size = step_size if step_size != 0 else 2
    # Set the cost function.
    cost_function = COST_FUNCTIONS[cost_function]
    # Get the macro-block from the current frame.
    macro_block = slice_macro_block(current_frame, x, y, width, height)
    # Set the starting point as the center of the macro-block.
    half_block_size = (width // 2, height // 2)
    cx, cy = x + half_block_size[0], y + half_block_size[1]
    # Keep the cost of (cx, cy), and update it and (cx, cy) if a block with smaller cost is found.
    min_cost = float('inf')
    while step_size > 1:
        # Calculate all points in all 4 directions.
        p1 = (cx, cy)
        p2 = (cx + step_size, cy)
        p3 = (cx - step_size, cy)
        p4 = (cx, cy + step_size)
        p5 = (cx, cy - step_size)
        for p in (p1, p2, p3, p4, p5):
            # Get the macro-block of the point from reference frame.
            ref_macro_block = slice_macro_block(reference_frame, p[0] - half_block_size[0], p[1] - half_block_size[1],
                                                width, height)
            # Calculate the cost between the blocks.
            cost = cost_function(macro_block, ref_macro_block)
            # Compare the cost to the minimum and update (cx, cy) and min_cost if a smaller cost was calculated.
            if cost < min_cost:
                min_cost = cost
                cx, cy = p
        # If the origin has the smallest cost, cut the step size by 2.
        if p1 == (cx, cy):
            step_size //= 2
    # When step_size is 1, check all 8 directions.
    p1 = (cx, cy)
    p2 = (cx + step_size, cy)
    p3 = (cx, cy + step_size)
    p4 = (cx + step_size, cy + step_size)
    p5 = (cx - step_size, cy)
    p6 = (cx, cy - step_size)
    p7 = (cx - step_size, cy - step_size)
    p8 = (cx + step_size, cy - step_size)
    p9 = (cx - step_size, cy + step_size)
    for p in (p1, p2, p3, p4, p5, p6, p7, p8, p9):
        # Get the macro-block of the point from reference frame.
        ref_macro_block = slice_macro_block(reference_frame, p[0] - half_block_size[0], p[1] - half_block_size[1],
                                            width, height)
        # Calculate the cost between the blocks.
        cost = cost_function(macro_block, ref_macro_block)
        # Compare the cost to the minimum and update (cx, cy) and min_cost if a smaller cost was calculated.
        if cost < min_cost:
            min_cost = cost
            cx, cy = p
    return cx - half_block_size[0], cy - half_block_size[1]


def diamond_search(current_frame: np.ndarray,
                   reference_frame: np.ndarray,
                   x: int,
                   y: int,
                   width: int,
                   height: int,
                   cost_function: str = 'MAD',
                   step_size: int = 2) -> Tuple[int, int]:
    """
    Diamond Search Algorithm for Block-Matching.

    :param current_frame: The current frame as a numpy array.
    :param reference_frame: The reference frame as a numpy array.
    :param x: The X-coordinate of the top-left point of a macro-block in the current frame to perform the search on.
    :param y: The Y-coordinate of the top-left point of a macro-block in the current frame to perform the search on.
    :param width: Width of the macro block.
    :param height: Height of the macro block.
    :param cost_function: The cost function to use.
    :param step_size: The distance of points to search from the origin search point.
    :return: The top-left point of the best macro-block in reference frame.
    """
    # Set step.
    step_size = step_size if step_size != 0 else 2
    # Set the cost function.
    cost_function = COST_FUNCTIONS[cost_function]
    # Get the macro-block from the current frame.
    macro_block = slice_macro_block(current_frame, x, y, width, height)
    # Set the starting point as the center of the macro-block.
    half_block_size = (width // 2, height // 2)
    cx, cy = x + half_block_size[0], y + half_block_size[1]
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
            ref_macro_block = slice_macro_block(reference_frame, p[0] - half_block_size[0], p[1] - half_block_size[1],
                                                width, height)
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
        ref_macro_block = slice_macro_block(reference_frame, p[0] - half_block_size[0], p[1] - half_block_size[1],
                                            width, height)
        # Calculate the cost between the blocks.
        cost = cost_function(macro_block, ref_macro_block)
        # Compare the cost to the minimum and update (cx, cy) and min_cost if a smaller cost was calculated.
        if cost < min_cost:
            min_cost = cost
            cx, cy = p
    return cx - half_block_size[0], cy - half_block_size[1]


SEARCH_FUNCTIONS = {
    'TSS':  three_step_search,
    'TDLS': two_dimensional_logarithmic_search,
    'DS':   diamond_search
}

from .cost_functions import *
from .utils import *


def tts(current_frame, reference_frame, x, y, block_size, cost_function=mad, steps=4):
    macro_block = slice_macro_block(current_frame, x, y, block_size)
    half_block_size = block_size // 2
    cx, cy = x + half_block_size, y + half_block_size
    min_cost = float('inf')
    while steps >= 1:
        p1 = (cx, cy)
        p2 = (cx + steps, cy)
        p3 = (cx, cy + steps)
        p4 = (cx + steps, cy + steps)
        p5 = (cx - steps, cy)
        p6 = (cx, cy - steps)
        p7 = (cx - steps, cy - steps)
        p8 = (cx + steps, cy - steps)
        p9 = (cx - steps, cy + steps)
        for p in (p1, p2, p3, p4, p5, p6, p7, p8, p9):
            ref_macro_block = slice_macro_block(reference_frame, p[0] - half_block_size, p[1] - half_block_size,
                                                block_size)
            cost = cost_function(ref_macro_block, macro_block)
            if cost < min_cost:
                min_cost = cost
                cx, cy = p
        steps //= 2
    return cx, cy

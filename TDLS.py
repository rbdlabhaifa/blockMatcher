import numpy as np

import utils


def TDLS(reference_frame, mc, macroblock_size, step):
    """
    :param reference_frame: of type numpy ndarray in BGR format
    :param mc: ndarray containing the top left coordinates of the macroblock to search for, formated as [x, y]
    :param macroblock_size: length of macroblock's size
    :param step: search step size of type int
    :return: returns absolute coordinates of the center of the block in pixels
    """
    mc_center = np.array([mc[0] - macroblock_size // 2, mc[1] - macroblock_size // 2])
    cur_block = utils.get_macro_block(mc[0], mc[1], reference_frame, macroblock_size)

    while True:
        p1 = [mc_center[0] + macroblock_size + step, mc_center[1]]
        p2 = [mc_center[0] - macroblock_size - step, mc_center[1]]
        p3 = [mc_center[0], mc_center[1] + macroblock_size + step]
        p4 = [mc_center[0], mc_center[1] - macroblock_size - step]

        points = [mc_center, p1, p2, p3, p4]

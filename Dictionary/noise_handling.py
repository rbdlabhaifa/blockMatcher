import numpy as np
from typing import Tuple, List
from block_matching.cost_functions import sad


def set_to_origin(motion_vectors: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
    """
    Fixes the tail of the vectors to the origin.

    :param motion_vectors: A list of tuples containing the start and end points of the vectors.
    :return: The new list of vectors.
    """
    return [(x2 - x1, y2 - y1) for x1, y1, x2, y2 in motion_vectors]


def evaluate_vector(vector: Tuple[int, int], reference_frame: np.ndarray, current_frame: np.ndarray) -> float:
    """
    A metric for comparing the quality of motion vectors that is calculated by shifting the reference frame and
    comparing it with the current frame.

    :param vector: The motion vector with its tail being the origin.
    :param reference_frame: The reference frame as a numpy array.
    :param current_frame: The current frame as a numpy array.
    :return: The quality of a motion vector as a float.
    """
    dx, dy = vector
    # Shift the reference frame and cut the rectangle that's in both the current frame and the shifted reference frame.
    if dy > 0:
        shifted_ref_frame = reference_frame[:-dy]
        shifted_cur_frame = current_frame[dy:]
    elif dy < 0:
        shifted_ref_frame = reference_frame[-dy:]
        shifted_cur_frame = current_frame[:dy]
    else:
        shifted_ref_frame = reference_frame
        shifted_cur_frame = current_frame
    if dx > 0:
        shifted_ref_frame = shifted_ref_frame[:, :-dx]
        shifted_cur_frame = shifted_cur_frame[:, dx:]
    elif dx < 0:
        shifted_ref_frame = shifted_ref_frame[:, -dx:]
        shifted_cur_frame = shifted_cur_frame[:, :dx]
    # Return the SAD to area ratio of the rectangle.
    return sad(shifted_ref_frame, shifted_cur_frame) / (shifted_ref_frame.shape[0] * shifted_ref_frame.shape[1])


def minimize_slope(motion_vectors: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Filter a list of motion vectors by slope.

    :param motion_vectors: A list of motion vectors.
    :return: A list of motion vectors that have the minimum slope.
    """
    min_slope = motion_vectors[0][1] / motion_vectors[0][0]
    filtered_vectors = [motion_vectors[0]]
    for i in range(1, len(motion_vectors)):
        slope = motion_vectors[i][1] / motion_vectors[i][0]
        if slope < min_slope:
            min_slope = slope
            filtered_vectors = [motion_vectors[i]]
        elif slope == min_slope:
            filtered_vectors.append(motion_vectors[i])
    return filtered_vectors

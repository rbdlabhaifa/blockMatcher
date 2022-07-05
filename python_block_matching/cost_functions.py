import numpy as np


def mad(current_macro_block: np.ndarray, reference_frame_block: np.ndarray) -> float:
    """
    Mean Absolute Difference (MAD).

    :param current_macro_block: A macro-block from the current frame which is a Numpy array containing RGB triples.
    :param reference_frame_block: A macro-block from the reference frame which is a Numpy array containing RGB triples.
    :return: A float used as a metric for evaluating a macro-block with another macro-block.
    """
    block_area = current_macro_block.shape[0] * current_macro_block.shape[1]
    return np.sum(np.abs(np.subtract(current_macro_block, reference_frame_block))) / block_area


def mse(current_macro_block: np.ndarray, reference_frame_block: np.ndarray) -> float:
    """
    Mean Squared Error (MSE).

    :param current_macro_block: A macro-block from the current frame which is a Numpy array containing RGB triples.
    :param reference_frame_block: A macro-block from the reference frame which is a Numpy array containing RGB triples.
    :return: A float used as a metric for evaluating a macro-block with another macro-block.
    """
    block_area = current_macro_block.shape[0] * current_macro_block.shape[1]
    return np.sum(np.square(np.subtract(current_macro_block, reference_frame_block))) / block_area


def sse(current_macro_block: np.ndarray, reference_frame_block: np.ndarray) -> float:
    """
    Sum of Squared Errors (SSE).

    :param current_macro_block: A macro-block from the current frame which is a Numpy array containing RGB triples.
    :param reference_frame_block: A macro-block from the reference frame which is a Numpy array containing RGB triples.
    :return: A float used as a metric for evaluating a macro-block with another macro-block.
    """
    block_area = current_macro_block.shape[0] * current_macro_block.shape[1]
    return np.sum(np.square(np.subtract(current_macro_block, reference_frame_block)))


def sad(current_macro_block: np.ndarray, reference_frame_block: np.ndarray) -> float:
    """
    Sum of Absolute Difference (SAD).

    :param current_macro_block: A macro-block from the current frame which is a Numpy array containing RGB triples.
    :param reference_frame_block: A macro-block from the reference frame which is a Numpy array containing RGB triples.
    :return: A float used as a metric for evaluating a macro-block with another macro-block.
    """
    return np.sum(np.abs(np.subtract(current_macro_block, reference_frame_block)))

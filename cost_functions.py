import numpy as np


def mad(current_macro_block: np.ndarray, reference_frame_block: np.ndarray, macro_block_size: int) -> float:
    """
    Mean Absolute Difference (MAD).

    :param current_macro_block: A macro-block from the current frame which is a Numpy array containing RGB triples.
    :param reference_frame_block: A macro-block from the reference frame which is a Numpy array containing RGB triples.
    :param macro_block_size: An int representing the size (width/high) of the macro-blocks.
    :return: A float used as a metric for evaluating a macro-block with another macro-block.
    """
    differences = np.abs(current_macro_block - reference_frame_block)
    return np.sum(differences) / (macro_block_size * macro_block_size)


def mse(current_macro_block: np.ndarray, reference_frame_block: np.ndarray, macro_block_size: int) -> float:
    """
    Mean Squared Error (MSE).

    :param current_macro_block: A macro-block from the current frame which is a Numpy array containing RGB triples.
    :param reference_frame_block: A macro-block from the reference frame which is a Numpy array containing RGB triples.
    :param macro_block_size: An int representing the size (width/height) of the macro-blocks.
    :return: A float used as a metric for evaluating a macro-block with another macro-block.
    """
    differences = np.square(current_macro_block - reference_frame_block)
    return np.sum(differences) / (macro_block_size * macro_block_size)

import numpy as np


def mad(current_frame_macro_block: np.ndarray, reference_frame_macro_block: np.ndarray) -> float:
    """
    Mean Absolute Difference (MAD).

    :param current_frame_macro_block: A slice from the current frame as a Numpy array of shape (y, x, 3).
    :param reference_frame_macro_block: A slice from the reference frame as a Numpy array of shape (y, x, 3).
    :return: A float used as a metric for evaluating the difference between two macro-blocks.
    """
    assert current_frame_macro_block.shape == reference_frame_macro_block.shape
    assert current_frame_macro_block.shape[2] == 3
    # Get the average of every RGB triple to transform the arrays to shape (x, y).
    current_frame_macro_block = np.sum(current_frame_macro_block, axis=2) / 3
    reference_frame_macro_block = np.sum(reference_frame_macro_block, axis=2) / 3
    # Calculate the area of the macro-blocks.
    block_area = current_frame_macro_block.shape[0] * current_frame_macro_block.shape[1]
    # Calculate the "MAD" of the arrays.
    return np.sum(np.abs(current_frame_macro_block - reference_frame_macro_block)) / block_area


def mse(current_frame_macro_block: np.ndarray, reference_frame_macro_block: np.ndarray) -> float:
    """
    Mean Squared Errors (MSE).

    :param current_frame_macro_block: A slice from the current frame as a Numpy array of shape (y, x, 3).
    :param reference_frame_macro_block: A slice from the reference frame as a Numpy array of shape (y, x, 3).
    :return: A float used as a metric for evaluating the difference between two macro-blocks.
    """
    assert current_frame_macro_block.shape == reference_frame_macro_block.shape
    assert current_frame_macro_block.shape[2] == 3
    # Get the average of every RGB triple to transform the arrays to shape (x, y).
    current_frame_macro_block = np.sum(current_frame_macro_block, axis=2) / 3
    reference_frame_macro_block = np.sum(reference_frame_macro_block, axis=2) / 3
    # Calculate the area of the macro-blocks.
    block_area = current_frame_macro_block.shape[0] * current_frame_macro_block.shape[1]
    # Calculate the "MSE" of the arrays.
    return np.sum(np.square(current_frame_macro_block - reference_frame_macro_block)) / block_area


def sse(current_frame_macro_block: np.ndarray, reference_frame_macro_block: np.ndarray) -> float:
    """
    Sum of Squared Error (SSE).

    :param current_frame_macro_block: A slice from the current frame as a Numpy array of shape (y, x, 3).
    :param reference_frame_macro_block: A slice from the reference frame as a Numpy array of shape (y, x, 3).
    :return: A float used as a metric for evaluating the difference between two macro-blocks.
    """
    assert current_frame_macro_block.shape == reference_frame_macro_block.shape
    assert current_frame_macro_block.shape[2] == 3
    # Get the average of every RGB triple to transform the arrays to shape (x, y).
    current_frame_macro_block = np.sum(current_frame_macro_block, axis=2) / 3
    reference_frame_macro_block = np.sum(reference_frame_macro_block, axis=2) / 3
    # Calculate the "SSE" of the arrays.
    return np.square(current_frame_macro_block - reference_frame_macro_block).sum()


def sad(current_frame_macro_block: np.ndarray, reference_frame_macro_block: np.ndarray) -> float:
    """
    Sum of Absolute Difference (SAD).

    :param current_frame_macro_block: A slice from the current frame as a Numpy array of shape (y, x, 3).
    :param reference_frame_macro_block: A slice from the reference frame as a Numpy array of shape (y, x, 3).
    :return: A float used as a metric for evaluating the difference between two macro-blocks.
    """
    assert current_frame_macro_block.shape == reference_frame_macro_block.shape
    assert current_frame_macro_block.shape[2] == 3
    # Get the average of every RGB triple to transform the arrays to shape (x, y).
    current_frame_macro_block = np.sum(current_frame_macro_block, axis=2) / 3
    reference_frame_macro_block = np.sum(reference_frame_macro_block, axis=2) / 3
    # Calculate the "SAD" of the arrays.
    return np.abs(current_frame_macro_block - reference_frame_macro_block).sum()


COST_FUNCTIONS = {
    'MAD': mad,
    'MSE': mse,
    'SSE': sse,
    'SAD': sad
}

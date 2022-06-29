import numpy as np
import cv2
from typing import Tuple, Union, List, Callable


def get_macro_blocks(frame: np.ndarray, size: int) -> Tuple[Tuple[int, int], np.ndarray]:
    """
    Returns a numpy array that contains the center points of all macro-blocks in the frame.

    :param frame: The frame as a numpy array.
    :param size: The size (width / height) of the macro-blocks.
    :return: A numpy array of all the center points of the macro-blocks.
    """
    for y in range(0, size * (frame.shape[0] // size), size):
        for x in range(0, size * (frame.shape[1] // size), size):
            yield (x, y), frame[y:y + size, x:x + size]


def slice_macro_block(frame: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    """
    Slice and return a macro-block from a frame.
    The macro-block's size is guaranteed, but there is no guarantee that (x, y) will be the top-left corner.

    :param frame: The frame to slice a block from.
    :param x: The X-coordinate of the top-left corner of the macro-block.
    :param y: The Y-coordinate of the top-left corner of the macro-block.
    :param size: The size of the macro-block.
    :return: The macro-block.
    """
    x, y = max(x, 0), max(y, 0)
    x, y = min(x, frame.shape[1] - size), min(y, frame.shape[0] - size)
    return frame[y:y + size, x:x + size]


def grayscale(frame: np.ndarray) -> np.ndarray:
    """
    Grayscales a frame.

    :param frame: A frame to grayscale.
    :return: The grayscaled frame as a numpy array.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def test_algorithm(source: Union[str, List[str]], algorithm: Callable, block_size: int = 16) -> None:
    """
    Tests an algorithm (from algorithms.py) or any function that gets 2 frames, a point and size, and returns a point.

    :param source: Either a string of the path of a video or a list that contains strings of paths to specific images.
    :param algorithm: The algorithm function to check or any function that gets matching parameters and returns a point.
    :param block_size: The size of the macro-blocks in the frame.
    :return: None
    """
    if isinstance(source, str):
        video = cv2.VideoCapture(source)
        read_successfully, reference_frame = video.read()
        gray_reference_frame = grayscale(reference_frame)
        while read_successfully:
            read_successfully, current_frame = video.read()
            if not read_successfully:
                break
            gray_current_frame = grayscale(current_frame)
            for (x, y), macro_block in get_macro_blocks(gray_current_frame, block_size):
                best_point = algorithm(gray_current_frame, gray_reference_frame, x, y, block_size)
                reference_frame = cv2.arrowedLine(reference_frame, (x, y), best_point, (100, 255, 50), 1)
            cv2.imshow('Press SPACE to continue.', reference_frame)
            cv2.waitKey(100000000)
            reference_frame = current_frame
            gray_reference_frame = gray_current_frame
    elif isinstance(source, list):
        images = []
        for path in source:
            if not isinstance(path, str):
                raise ValueError('Parameter \'source\' must be a list of strings or a string.')
            images.append(cv2.imread(path))
        while len(images) > 1:
            reference_frame = grayscale(images[0].copy())
            current_frame = grayscale(images[1].copy())
            for (x, y), macro_block in get_macro_blocks(current_frame, block_size):
                best_point = algorithm(current_frame, reference_frame, x, y, block_size)
                images[0] = cv2.arrowedLine(images[0], (x, y), best_point, (100, 255, 50), 1)
            cv2.imshow('Press SPACE to continue.', images[0])
            cv2.waitKey(100000000)
            images = images[1:]
    else:
        raise ValueError('Parameter \'source\' must be a list of strings or a string.')

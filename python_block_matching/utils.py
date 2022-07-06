import numpy as np
import cv2
from .cost_functions import *
from typing import Tuple, Union, List, Callable


def get_macro_blocks(frame: np.ndarray, size: int) -> Tuple[int, int]:
    """
    Returns a numpy array that contains the center points of all macro-blocks in the frame.

    :param frame: The frame as a numpy array.
    :param size: The size (width / height) of the macro-blocks.
    :return: A numpy array of all the center points of the macro-blocks.
    """
    for y in range(0, size * (frame.shape[0] // size), size):
        for x in range(0, size * (frame.shape[1] // size), size):
            yield x, y


def intra_frame_mb_partition(frame: np.ndarray, threshold, max_size=16):
    half_size = max_size // 2
    for tlx, tly in get_macro_blocks(frame, max_size):
        mb = slice_macro_block(frame, tlx, tly, max_size)
        v1, v2 = mb[:half_size, :], mb[half_size:, :]
        h1, h2 = mb[:, :half_size], mb[:, half_size:]
        vertical_cost = sad(v1, v2)
        horizontal_cost = sad(h1, h2)
        threshold = 5 * np.sum(mb)
        if vertical_cost > threshold and horizontal_cost > threshold:
            b1 = (tlx, tly, half_size, half_size)
            b2 = (tlx + half_size, tly, half_size, half_size)
            b3 = (tlx, tly + half_size, half_size, half_size)
            b4 = (tlx + half_size, tly + half_size, half_size, half_size)
            q_size = half_size // 2
            for b in (b1, b2, b3, b4):
                yield b
                # mb = slice_macro_block(frame, b[0], b[1], b[2])
                # v1, v2 = mb[:q_size, :], mb[q_size:, :]
                # h1, h2 = mb[:, :q_size], mb[:, q_size:]
                # vertical_cost = func(v1, v2)
                # horizontal_cost = func(h1, h2)
                # if vertical_cost > threshold and horizontal_cost > threshold:
                #     yield b[0], b[1], q_size, q_size
                #     yield b[0] + q_size, b[1], q_size, q_size
                #     yield b[0], b[1] + q_size, q_size, q_size
                #     yield b[0] + q_size, b[1] + q_size, q_size, q_size
                # elif vertical_cost > threshold:
                #     yield b[0], b[1], half_size, q_size
                #     yield b[0], b[1] + q_size, half_size, q_size
                # elif horizontal_cost > threshold:
                #     yield b[0], b[1], q_size, half_size
                #     yield b[0], b[1] + q_size, q_size, half_size
                # else:
                #     yield b
        elif vertical_cost > threshold:
            yield tlx, tly, half_size, max_size
            yield tlx + half_size, tly, half_size, max_size
        elif horizontal_cost > threshold:
            yield tlx, tly, max_size, half_size
            yield tlx, tly + half_size, max_size, half_size
        else:
            yield tlx, tly, max_size, max_size



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


def test_algorithm(source: Union[str, List[str]], algorithm: Callable, bsize: int = 16, cost_f: str = 'MAD') -> None:
    """
    Tests an algorithm (from algorithms.py) or any function that gets 2 frames, a point and size, and returns a point.

    :param source: Either a string of the path of a video or a list that contains strings of paths to specific images.
    :param algorithm: The algorithm function to check or any function that gets matching parameters and returns a point.
    :param bsize: The size of the macro-blocks in the frame.
    :param cost_f: The cost function to use, can only be either 'MAD' or 'MSE'.
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
            for (x, y), macro_block in get_macro_blocks(gray_current_frame, bsize):
                best_point = algorithm(gray_current_frame, gray_reference_frame, x, y, bsize, cost_f)
                p = (x + bsize // 2, y + bsize // 2)
                if p != best_point:
                    reference_frame = cv2.arrowedLine(reference_frame, best_point, p, (100, 255, 50), 1)
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
            for (x, y), macro_block in get_macro_blocks(current_frame, bsize):
                best_point = algorithm(current_frame, reference_frame, x, y, bsize, cost_f)
                p = (x + bsize // 2, y + bsize // 2)
                if p != best_point:
                    images[0] = cv2.arrowedLine(images[0], best_point, p, (100, 255, 50), 1)
            cv2.imshow('Press SPACE to continue.', images[0])
            cv2.waitKey(100000000)
            images = images[1:]
    else:
        raise ValueError('Parameter \'source\' must be a list of strings or a string.')

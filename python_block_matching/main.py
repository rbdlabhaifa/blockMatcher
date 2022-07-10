from typing import Tuple, Union, List
import numpy as np
import cv2


from .block_matching import SEARCH_FUNCTIONS
from .block_partitioning import PARTITIONING_FUNCTION


class BMFrame:

    def __init__(self, image: np.ndarray):
        self.base_image = image.copy()
        self.drawn_image = image.copy()

    def draw_macro_block(self, *blocks: Tuple[int, int, int, int], color: Tuple[int, int, int], thickness: int) -> None:
        """
        Draw a block on the frame.

        :param blocks: The macro-blocks to draw (top_left_x, top_left_y, width, height).
        :param color: The RGB color of the rectangle.
        :param thickness: The thickness of the rectangle.
        """
        for x, y, w, h in blocks:
            self.drawn_image = cv2.rectangle(self.drawn_image, (x, y), (x + w, y + h), color, thickness)

    def draw_motion_vector(self, *vectors: Tuple[int, int, int, int], color: Tuple[int, int, int],
                           thickness: int) -> None:
        """
        Draw a vector on the frame.

        :param vectors: The start and end of the vector (start_x, start_y, end_x, end_y).
        :param color: The RGB color of the arrow.
        :param thickness: The thickness of the arrow.
        """
        for x1, y1, x2, y2 in vectors:
            self.drawn_image = cv2.arrowedLine(self.drawn_image, (x1, y1), (x2, y2), color, thickness)

    def reset(self) -> None:
        """
        Reset all drawings on the frame.
        """
        self.drawn_image = self.base_image.copy()

    def show(self) -> None:
        """
        Show the frame on a window.
        """
        cv2.imshow('', self.drawn_image)
        cv2.waitKey()

    def __getitem__(self, item) -> np.ndarray:
        return self.base_image[item]

    def __str__(self) -> str:
        return str(self.base_image)


class BMVideo:

    def __init__(self, video_or_frames: Union[str, list]):
        assert isinstance(video_or_frames, str) or isinstance(video_or_frames, list)
        self.frames = video_or_frames

    def __getitem__(self, item) -> Union[BMFrame, List[BMFrame]]:
        if isinstance(item, slice):
            frames = []
            for i in range(item.start, item.stop, item.step):
                frames.append(self[i])
            return frames
        assert isinstance(item, int)
        if isinstance(self.frames, str):
            video = cv2.VideoCapture(self.frames)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            item = (frame_count + item) if item < 0 else item
            assert 0 <= item < frame_count
            _, frame = video.read()
            for i in range(item):
                _, frame = video.read()
            return BMFrame(frame)
        else:
            item = (len(self.frames) + item) if item < 0 else item
            assert 0 <= item < len(self.frames)
            return BMFrame(cv2.imread(self.frames[item]))

    def __str__(self) -> str:
        return str(self.frames)


class BlockMatching:

    @staticmethod
    def generate_motion_vectors(current_frame: np.ndarray, reference_frame: np.ndarray,
                                current_frame_blocks: List[Tuple[int, int, int, int]],
                                search_function: str, cost_function: str,
                                step_size: int = 0) -> Tuple[int, int, int, int]:
        """
        Generates motion vectors from the reference frame to the current frame.

        :param current_frame: The current frame as a numpy array.
        :param reference_frame: The reference frame as a numpy array.
        :param current_frame_blocks: The macro-blocks in the current frame.
        :param search_function: The function to search for the best macro-blocks with.
        :param cost_function: The cost function to use for the search.
        :param step_size: The step size to use in the search (0 to use the default of the search function).
        :return: The start and end points of the vectors.
        """
        search_function = SEARCH_FUNCTIONS[search_function]
        for x, y, w, h in current_frame_blocks:
            sx, sy = search_function(current_frame, reference_frame, x, y, w, h, cost_function, step_size)
            hw, hh = w // 2, h // 2
            yield sx + hw, sy + hh, x + hw, y + hh

    @staticmethod
    def extract_motion_vectors():
        pass

    @staticmethod
    def generate_macro_blocks(frame: np.ndarray, block_width: int, block_height: int,
                              partition_function: str, cost_function: str = 'SAD') -> Tuple[int, int, int, int]:
        """
        Generate macro-blocks in a frame.

        :param frame: The frame generate the macro-blocks for.
        :param block_width: The max width of the macro-blocks.
        :param block_height: The max height of the macro-blocks.
        :param partition_function: The function to partition the frame with.
        :param cost_function: The cost function to use for the partitioning function.
        :return: The macro-blocks top-left point and their width and height.
        """
        partition_function = PARTITIONING_FUNCTION[partition_function]
        for x, y, macro_block in partition_function(frame, block_width, block_height, cost_function):
            yield x, y, *macro_block.shape

    @staticmethod
    def extract_macro_blocks():
        pass

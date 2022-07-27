from typing import Tuple, Union, List, Dict
import numpy as np
import cv2


from .block_matching import SEARCH_FUNCTIONS
from .block_partitioning import PARTITIONING_FUNCTION


class BMFrame:

    def __init__(self, image: np.ndarray):
        self.base_image = image.copy()
        self.drawn_image = image.copy()

    def draw_macro_block(self, blocks: List[Tuple[int, int, int, int]], color: Tuple[int, int, int],
                         thickness: int) -> None:
        """
        Draw a block on the frame.

        :param blocks: The macro-blocks to draw (top_left_x, top_left_y, width, height).
        :param color: The BGR color of the rectangle.
        :param thickness: The thickness of the rectangle.
        """
        for x, y, w, h in blocks:
            self.drawn_image = cv2.rectangle(self.drawn_image, (x, y), (x + w, y + h), color, thickness)

    def draw_motion_vector(self, vectors: List[Tuple[int, int, int, int]], color: Tuple[int, int, int],
                           thickness: int) -> None:
        """
        Draw a vector on the frame.

        :param vectors: The start and end of the vector (start_x, start_y, end_x, end_y).
        :param color: The BGR color of the arrow.
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

    def get_frame_count(self) -> int:
        """
        Get the count of frames in this video.

        :return: The amount of frames.
        """
        if isinstance(self.frames, str):
            video = cv2.VideoCapture(self.frames)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < 0:
                frame_count = 1
                video_copy = cv2.VideoCapture(self.frames)
                read_successfully, frame = video_copy.read()
                while read_successfully:
                    read_successfully, frame = video_copy.read()
                    frame_count += 1
            return frame_count
        return len(self.frames)

    def __getitem__(self, item) -> Union[BMFrame, List[BMFrame]]:
        if isinstance(item, slice):
            frames = []
            for i in range(item.start, item.stop, item.step):
                frames.append(self[i])
            return frames
        assert isinstance(item, int)
        if isinstance(self.frames, str):
            video = cv2.VideoCapture(self.frames)
            frame_count = self.get_frame_count()
            item = (frame_count + item) if item < 0 else item
            # assert 0 <= item < frame_count
            _, frame = video.read()
            for i in range(item - 1):
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
    def get_motion_vectors(current_frame: np.ndarray, reference_frame: np.ndarray,
                           search_function: str = 'DS', cost_function: str = 'SAD', step_size: int = 0,
                           current_frame_mb: List[Tuple[int, int, int, int]] = None) -> List[Tuple[int, int, int, int]]:
        """
        Generates motion vectors from the reference frame to the current frame.

        :param current_frame: The current frame as a numpy array.
        :param reference_frame: The reference frame as a numpy array.
        :param search_function: The function to search for the best macro-blocks with.
        :param cost_function: The cost function to use for the search.
        :param step_size: The step size to use in the search (0 to use the default of the search function).
        :param current_frame_mb: The macro-blocks int the current frame to run the search on.
        :return: The start and end points of the vectors.
        """
        if current_frame_mb is None:
            current_frame_mb = BlockMatching.get_macro_blocks(current_frame)
        motion_vectors = []
        search_function = SEARCH_FUNCTIONS[search_function]
        for x, y, w, h in current_frame_mb:
            sx, sy = search_function(current_frame, reference_frame, x, y, w, h, cost_function, step_size)
            hw, hh = w // 2, h // 2
            motion_vectors.append((sx + hw, sy + hh, x + hw, y + hh))
        return motion_vectors

    @staticmethod
    def extract_motion_vectors(file_path: str) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """
        Gets a file that was generated using extract_mvs.c and returns the motion vectors for each frame.

        :param file_path: The path for the file.
        :return: A dictionary where the frame's index is the key and a list containing the motion-vectors is the value.
        """
        motion_vectors_by_frame = {0: []}
        with open(file_path) as file:
            file.readline()
            for line in file:
                frame_num, _, _, _, x2, y2, x1, y1, _ = eval(line)
                if frame_num != len(motion_vectors_by_frame):
                    if frame_num - 1 > len(motion_vectors_by_frame):
                        motion_vectors_by_frame[frame_num - 2] = []
                    else:
                        motion_vectors_by_frame[frame_num - 1] = [(x1, y1, x2, y2)]
                else:
                    motion_vectors_by_frame[frame_num - 1].append((x1, y1, x2, y2))
        return motion_vectors_by_frame

    @staticmethod
    def get_macro_blocks(frame: np.ndarray, block_width: int = 16, block_height: int = 16,
                         partition_function: str = 'FIXED',
                         cost_function: str = 'SAD') -> List[Tuple[int, int, int, int]]:
        """
        Generate macro-blocks in a frame.

        :param frame: The frame generate the macro-blocks for.
        :param block_width: The max width of the macro-blocks.
        :param block_height: The max height of the macro-blocks.
        :param partition_function: The function to partition the frame with.
        :param cost_function: The cost function to use for the partitioning function.
        :return: A list that contains the top-left point of the macro-blocks and their width and height.
        """
        macro_blocks = []
        partition_function = PARTITIONING_FUNCTION[partition_function]
        for x, y, w, h in partition_function(frame, block_width, block_height, cost_function):
            macro_blocks.append((x, y, w, h))
        return macro_blocks

    @staticmethod
    def extract_macro_blocks(file_path: str) -> Dict[int, List[Tuple[int, int, int, int]]]:
        """
        Gets a file that was generated using extract_mvs.c and returns the macro-blocks for each frame.

        :param file_path: The path for the file.
        :return: A dictionary where the frame's index is the key and a list containing the macro-blocks is the value.
        """
        macro_blocks_by_frame = {0: []}
        with open(file_path) as file:
            file.readline()
            for line in file:
                frame_num, _, block_w, block_h, _, _, x, y, _ = eval(line)
                x -= block_w // 2
                y -= block_h // 2
                if frame_num != len(macro_blocks_by_frame):
                    if frame_num - 1 > len(macro_blocks_by_frame):
                        macro_blocks_by_frame[frame_num - 2] = []
                    else:
                        macro_blocks_by_frame[frame_num - 1] = [(x, y, block_w, block_h)]
                else:
                    macro_blocks_by_frame[frame_num - 1].append((x, y, block_w, block_h))
        return macro_blocks_by_frame

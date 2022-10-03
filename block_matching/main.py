from typing import Tuple, List, Union
import numpy as np
import subprocess
import os
import tempfile
import cv2
import shutil

from .block_matching import SEARCH_FUNCTIONS
from .block_partitioning import PARTITIONING_FUNCTION


class BlockMatching:

    # ============================================= Extract Motion Data ============================================= #

    MOTION_VECTORS_EXECUTABLE_PATH = os.getcwd() + r'/extra/extract motion data/motionVectors'

    @staticmethod
    def extract_motion_data(video_path: str, extract_path: str = None) -> List[List[Tuple[int, int, int, int]]]:
        """
        Extracts the motion data of a video.

        :param video_path: The path to the video.
        :param extract_path: The path to the motionVectors executable, if None uses default path.
        :return: A list that contains lists of motion vectors between every two consecutive frames.
        """
        if extract_path is None:
            extract_path = BlockMatching.MOTION_VECTORS_EXECUTABLE_PATH
        motion_data = subprocess.run([extract_path, video_path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        motion_data = motion_data.stdout.decode().strip().split('\n')
        frames_vectors = []
        for i in range(1, len(motion_data)):
            frame_num, _, block_width, block_height, src_x, src_y, dst_x, dst_y, _ = eval(motion_data[i])
            while frame_num - 1 != len(frames_vectors):
                frames_vectors.append([])
            frames_vectors[frame_num - 2].append((src_x, src_y, dst_x, dst_y))
        return frames_vectors

    # ============================================= Calculate Motion Data =========================================== #

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
    def get_ffmpeg_motion_vectors(frames: List[Union[str, np.ndarray]], save_to: str = None,
                                  extract_path: str = None, on_raspi: bool = False,
                                  image_format: str = 'png',
                                  repeat_first_frame: bool = True) -> List[List[Tuple[int, int, int, int]]]:
        """
        Generates motion vectors from the first frame to the rest of the frames with ffmpeg.

        :param frames: A list of frames. Either the paths to the frames or the frames as arrays.
        :param save_to: The path to save the video to. If None deletes the video.
        :param extract_path: The path to the motionVectors executable, if None uses default path.
        :param on_raspi: Set to True if the code is running on a raspberry pi to use hardware encoding.
        :param image_format: The format of the images (png, jpg, ...).
        :param repeat_first_frame: If true every even frame is the first frame.
        :return: A list that contains lists of motion vectors.
        """
        temporary_directory = tempfile.mkdtemp()
        try:
            base_frame = frames[0]
            if isinstance(base_frame, str):
                base_frame = cv2.imread(base_frame)
            frame_index = 0
            if not repeat_first_frame:
                cv2.imwrite(temporary_directory + f'/{frame_index}.{image_format}', base_frame)
            for frame in frames[1:]:
                if repeat_first_frame:
                    cv2.imwrite(temporary_directory + f'/{frame_index}.{image_format}', base_frame)
                    frame_index += 1
                if isinstance(frame, str):
                    shutil.copyfile(frame, temporary_directory + f'/{frame_index}.{image_format}')
                else:
                    cv2.imwrite(temporary_directory + f'/{frame_index}.{image_format}', frame)
                frame_index += 1
            if on_raspi:
                subprocess.run(['ffmpeg', '-i', f'%d.{image_format}', '-input_format', 'yuv420p', '-pix_fmt', 'yuv420p',
                                '-b:v', '320M', '-c:v', 'h264_v4l2m2m', 'out.h264'], cwd=temporary_directory)
                if save_to is not None:
                    shutil.copyfile(temporary_directory + f'/out.h264', save_to)
                motion_data = BlockMatching.extract_motion_data(temporary_directory + '/out.h264', extract_path)
            else:
                subprocess.run(['ffmpeg', '-i', f'%d.{image_format}', '-input_format', 'yuv420p', '-c:v', 'h264',
                                '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', 'out.mp4'], cwd=temporary_directory)
                if save_to is not None:
                    shutil.copyfile(temporary_directory + f'/out.mp4', save_to)
                motion_data = BlockMatching.extract_motion_data(temporary_directory + '/out.mp4', extract_path)
            shutil.rmtree(temporary_directory, ignore_errors=True)
            return motion_data
        except (OSError, Exception) as error:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            raise error

    @staticmethod
    def get_opencv_motion_vectors(current_frame: Union[str, np.ndarray], reference_frame: Union[str, np.ndarray],
                                  extract_path: str = None) -> List[Tuple[int, int, int, int]]:
        """
        Generates motion vectors from the reference frame to the current frame using opencv.

        :param current_frame: Either a path to an image of the current frame or the current frame as an array.
        :param reference_frame: Either a path to an image of the reference frame or the reference frame an array.
        :param extract_path: The path to the motionVectors executable, if None uses default path.
        :return: A list that contains the motion vectors from the reference frame to the current frame.
        """
        temporary_file = tempfile.NamedTemporaryFile(suffix='.mp4')
        try:
            if isinstance(current_frame, str):
                current_frame = cv2.imread(current_frame)
            if isinstance(reference_frame, str):
                reference_frame = cv2.imread(reference_frame)
            writer = cv2.VideoWriter(temporary_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 10,
                                     (current_frame.shape[1], current_frame.shape[0]))
            writer.write(reference_frame)
            writer.write(current_frame)
            writer.release()
            motion_data = BlockMatching.extract_motion_data(temporary_file.name, extract_path)
            temporary_file.close()
            return motion_data[0]
        except (OSError, Exception) as error:
            temporary_file.close()
            raise error

    # ============================================= View Motion Data ================================================ #

    @staticmethod
    def draw_motion_vectors(frame: np.ndarray, motion_vectors: List[Tuple[int, int, int, int]],
                            color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 1,
                            save_to: str = None) -> np.ndarray:
        """
        Draws motion vectors on a frame.

        :param frame: The frame as a numpy array.
        :param motion_vectors: A list of motion vectors.
        :param color: The color to draw the motion vectors with.
        :param thickness: The thickness of the vectors.
        :param save_to: The path to save the drawn frame to, if None doesn't save.
        :return: The drawn frame.
        """
        frame = frame.copy()
        for x1, y1, x2, y2 in motion_vectors:
            frame = cv2.arrowedLine(frame, (x1, y1), (x2, y2), color, thickness)
        if save_to is not None:
            cv2.imwrite(save_to, frame)
        return frame

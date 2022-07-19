import cv2
import numpy as np
from typing import Union, List
from PIL import Image
from python_block_matching import BMFrame, BlockMatching


class DataGenerator:

    @staticmethod
    def get_gradient_2d(start, stop, width, height, is_horizontal):
        if is_horizontal:
            return np.tile(np.linspace(start, stop, width), (height, 1))
        else:
            return np.tile(np.linspace(start, stop, height), (width, 1)).T

    @staticmethod
    def generate_translation(frame_size: List[int], img_url: str, translation):
        """
        @param frame_size: [width, height] of overall frame, must be bigger than the pasted image's size
        @param img_url: url of the image to be pasted.
        @param translation: [translation in the x-axis, translation in the y-axis]
        @return: The frame before the motion (Reference), The frame after the motion (Current)
        """
        frame_size = frame_size.copy()
        frame_size.append(frame_size.pop(0))
        # frame_size.append(3)
        ref_frame = Image.open("synthetic data/gradient.jpeg")
        ref_frame = ref_frame.resize(frame_size)
        cur_frame = ref_frame.copy()
        img = Image.open(img_url)
        # assert ref_frame.size[0] > img.size[0] and ref_frame.size[1] > img.size[1]
        ref_frame.paste(img, (ref_frame.size[0] // 2 - img.size[0] // 2, ref_frame.size[1] // 2 - img.size[1] // 2,
                              ref_frame.size[0] // 2 + img.size[0] // 2, ref_frame.size[1] // 2 + img.size[1] // 2))
        cur_frame.paste(img, (
            cur_frame.size[0] // 2 - img.size[0] // 2 + translation[0],
            cur_frame.size[1] // 2 - img.size[1] // 2 + translation[1],
            cur_frame.size[0] // 2 + img.size[0] // 2 + translation[0],
            cur_frame.size[1] // 2 + img.size[1] // 2 + translation[1]))

        return np.asarray(ref_frame), np.asarray(cur_frame)

    @staticmethod
    def generate_rotation(frame_size: List[int], img_url: str, angle: Union[int, float]):
        """
        @param frame_size: [width, height] of overall frame, must be bigger than the pasted image's size
        @param img_url: url of the image to be pasted.
        @param angle: angle of rotation in degrees. positive = clockwise, negative = counter-clockwise
        @return: The frame before the motion (Reference), The frame after the motion (Current)
        """
        frame_size = frame_size.copy()
        frame_size.append(frame_size.pop(0))
        # frame_size.append(3)
        ref_frame = Image.open("synthetic data/gradient.jpeg")
        ref_frame = ref_frame.resize(frame_size)
        cur_frame = ref_frame.copy()
        img = Image.open(img_url)
        assert ref_frame.size[0] > img.size[0] and ref_frame.size[1] > img.size[1]
        ref_frame.paste(img, (ref_frame.size[0] // 2 - img.size[0] // 2, ref_frame.size[1] // 2 - img.size[1] // 2,
                              ref_frame.size[0] // 2 + img.size[0] // 2, ref_frame.size[1] // 2 + img.size[1] // 2))
        img = img.convert("RGBA")
        img = img.rotate(-angle, expand=True)
        cur_frame.paste(img, (ref_frame.size[0] // 2 - img.size[0] // 2, ref_frame.size[1] // 2 - img.size[1] // 2,
                              ref_frame.size[0] // 2 + img.size[0] // 2, ref_frame.size[1] // 2 + img.size[1] // 2),
                        img)

        return np.asarray(ref_frame), np.asarray(cur_frame)


if __name__ == '__main__':
    f1, f2 = DataGenerator.generate_rotation([360, 360], "synthetic data/Image0.png", 30)
    f2 = BMFrame(f2)
    f2.show()

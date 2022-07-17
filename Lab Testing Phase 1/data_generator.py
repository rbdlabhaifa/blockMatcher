import cv2
import numpy as np
from typing import Tuple, Union, List, Dict
from PIL import Image, ImageDraw

from python_block_matching import BMFrame, BlockMatching


class DataGenerator:
    @staticmethod
    def generate_translation(frame_size: List[int], img_url: str, translation):
        """
        @param frame_size: [width, height] of overall frame, must be bigger than the pasted image's size
        @param img_url: url of the image to be pasted.
        @param translation: [translation in the x-axis, translation in the y-axis]
        @return: The frame before the motion (Reference), The frame after the motion (Current)
        """
        frame_size.append(frame_size.pop(0))
        frame_size.append(3)
        ref_frame = Image.fromarray(np.full(frame_size, 255, dtype=np.uint8), 'RGB')
        cur_frame = ref_frame.copy()
        img = Image.open(img_url)
        assert ref_frame.size[0] > img.size[0] and ref_frame.size[1] > img.size[1]
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
        frame_size.append(frame_size.pop(0))
        frame_size.append(3)
        ref_frame = Image.fromarray(np.full(frame_size, 255, dtype=np.uint8), 'RGB')
        cur_frame = ref_frame.copy()
        img = Image.open(img_url)
        assert ref_frame.size[0] > img.size[0] and ref_frame.size[1] > img.size[1]
        ref_frame.paste(img, (ref_frame.size[0] // 2 - img.size[0] // 2, ref_frame.size[1] // 2 - img.size[1] // 2,
                              ref_frame.size[0] // 2 + img.size[0] // 2, ref_frame.size[1] // 2 + img.size[1] // 2))
        img = img.rotate(-angle, expand=True, fillcolor=(255, 255, 255))
        cur_frame.paste(img, (ref_frame.size[0] // 2 - img.size[0] // 2, ref_frame.size[1] // 2 - img.size[1] // 2,
                              ref_frame.size[0] // 2 + img.size[0] // 2, ref_frame.size[1] // 2 + img.size[1] // 2))

        return np.asarray(ref_frame), np.asarray(cur_frame)


if __name__ == '__main__':
    image = np.full((160, 160, 3), 255, dtype=np.uint8)

    cv2.circle(img=image, center=(80, 80), radius=80, color=(0, 255, 0), thickness=-1)

    # cv2.imwrite("Benchmark_Pictures/Circle_full.png", image)
    f1, f2 = DataGenerator.generate_translation([360, 360], "Benchmark_Pictures/Circle_full.png", [40, 0])
    f1 = BMFrame(f1)
    f2 = BMFrame(f2)
    mv = list(BlockMatching.get_motion_vectors(f2.base_image, f1.base_image))
    f1.draw_motion_vector(mv, (255, 0, 0), 1)
    while True:
        f1.show()
        f2.show()

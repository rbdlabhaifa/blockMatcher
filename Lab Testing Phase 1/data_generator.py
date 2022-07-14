import cv2
import numpy as np
from typing import Tuple, Union, List, Dict
from PIL import Image

from python_block_matching import BMFrame


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
        img = img.rotate(angle, expand=True, fillcolor=(255,255,255))
        cur_frame.paste(img, (ref_frame.size[0] // 2 - img.size[0] // 2, ref_frame.size[1] // 2 - img.size[1] // 2,
                              ref_frame.size[0] // 2 + img.size[0] // 2, ref_frame.size[1] // 2 + img.size[1] // 2))

        return np.asarray(ref_frame), np.asarray(cur_frame)


if __name__ == '__main__':
    ref, cur = DataGenerator.generate_rotation([360, 360],
                                               "/home/txp2/RPI-BMA-RE/Lab Testing Phase 1/Benchmark_Pictures/Image0.png",
                                               20)
    ref = BMFrame(ref)
    cur = BMFrame(cur)
    ref.show()
    cur.show()
    cv2.destroyAllWindows()

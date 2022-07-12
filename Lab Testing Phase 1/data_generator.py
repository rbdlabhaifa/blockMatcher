import cv2
import numpy as np
from typing import List
from PIL import Image


class DataGenerator:
    @staticmethod
    def generate_movement(frame_size: List[int], img_url: str, translation):
        """
        @param frame_size: [width, height] of overall frame, must be bigger than the pasted image's size
        @param img_url: url of the image to be pasted.
        @param translation: [translation in the x-axis, translation in the y-axis]
        @return:
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

if __name__ == '__main__':
    # for i in range(15):
    arr = np.full((10,10,3),[0,0,0], dtype=np.uint8)
    # arr = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    arr = cv2.resize(arr, (160, 160), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(arr, "RGB")
    img.save("/home/txp2/RPI-BMA-RE/Lab Testing Phase 1/Benchmark_Pictures/Black.png")

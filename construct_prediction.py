import numpy as np

from python_block_matching.utils import *
from python_block_matching import algorithms
import numpy
import cv2

ref_url = "frame1.png"
cur_url = "frame2.png"

ref_img = cv2.imread(ref_url)
cur_img = cv2.imread(cur_url)
prediction_img = numpy.empty(cur_img.shape[0], dtype=np.ndarray)
print(cur_img.shape)

row = np.empty((16, 16, 3))
last_y = 0
for x, y in get_macro_blocks(cur_img, 16):
    if y == last_y:
        cords = algorithms.three_step_search(cur_img, ref_img, x, y, 16)
        row = np.concatenate(slice_macro_block(ref_img, cords[0] - 8, cords[1] - 8, 16))
    else:
        cv2.imshow("", row)
        cv2.waitKey()
        row = np.empty((16, 16, 3))
    last_y = y

# blocks = np.array(blocks).reshape((960, 1280, 3))
# print(blocks.shape)
# cv2.imshow("pred", blocks)
# cv2.waitKey()

import cv2

from python_block_matching.algorithms import get_macro_block,three_step_search
from python_block_matching.cost_functions import mad
from python_block_matching.utils import *

ref_image = "/home/txp2/RPI-BMA-RE/frame10.png"
cur_image = "/home/txp2/RPI-BMA-RE/frame11.png"
cur_image = cv2.imread(cur_image)
ref_image = cv2.imread(ref_image)
w, h = cur_image.shape[0], cur_image.shape[1]
ref_search_area = get_macro_block(w // 2 - 12, h // 2 - 12, ref_image, 24)
cur_block = get_macro_block(w // 2 - 8, h // 2 - 8, cur_image, 16)
search = three_step_search(cur_block, ref_search_area, 16, mad)
search = search[0] + w // 2, search[1] + h // 2
print(search, (w // 2, h // 2))
ref_image = cv2.arrowedLine(ref_image, (ref_image.shape[0] // 2, ref_image.shape[1] // 2), search, (255, 0, 0), 3)
cv2.imshow('cum', ref_image)
cv2.waitKey(123999)

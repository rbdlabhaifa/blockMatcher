import cv2

from algorithms import *
from utils import *

ref_image = "/home/txp2/RPI-BMA-RE/frame10.png"
cur_image = "/home/txp2/RPI-BMA-RE/frame11.png"
cur_image = cv2.imread(cur_image)
ref_image = cv2.imread(ref_image)
w, h = cur_image.shape[0], cur_image.shape[1]
ref_search_area = get_macro_block(w // 2, h // 2, ref_image, 23)
cur_block = get_macro_block(w // 2, h // 2, cur_image, 16)
search = three_step_search(cur_block, ref_search_area, 16, mad)
ref_image = cv2.arrowedLine(ref_image, (ref_image.shape[0] // 2, ref_image.shape[1] // 2), search, (255, 0, 0), 3)
cv2.imshow('cum', ref_image)

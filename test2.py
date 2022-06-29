from python_block_matching.utils import *
from python_block_matching.cost_functions import *
from python_block_matching.algorithms import *
import cv2


frame1, frame2 = cv2.imread('frame1.png'), cv2.imread('frame2.png')
image = cv2.imread('frame1.png')

for p, block in get_macro_blocks(frame2, 16):
    best_point = three_step_search(frame1, frame2, p[0], p[1], 16, 'MAD')
    image = cv2.arrowedLine(image, best_point, p, (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(100000)
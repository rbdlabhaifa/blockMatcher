from python_block_matching.utils import *
from python_block_matching.cost_functions import *
from python_block_matching.algorithms import *
import cv2


frame1, frame2 = cv2.imread('frame1.png'), cv2.imread('frame2.png')
frame = frame1.copy()
for (x, y), block in get_macro_blocks(frame2, 16):
    bp = three_step_search(frame2, frame2, x, y, 16)
    frame = cv2.arrowedLine(frame, bp, (x, y), (0, 255, 0), 1)
cv2.imshow('bo', frame)
cv2.waitKey(210000)



# vid = cv2.VideoCapture('vid.mp4')
# suc, frame1 = vid.read()
# frame2 = frame1.copy()
# image = frame2.copy()
# for p, block in get_macro_blocks(frame2, 16):
#     best_point = three_step_search(frame1, frame2, p[0], p[1], 16, 'MAD')
#     image = cv2.arrowedLine(image, p, best_point, (0, 255, 0), 1)
#
# cv2.imshow('press space', image)
# cv2.waitKey(100000)
# suc, frame1 = vid.read()
#
# while suc:
#     suc, frame2 = vid.read()
#     image = frame2.copy()
#
#     for p, block in get_macro_blocks(frame2, 16):
#         best_point = three_step_search(frame1, frame2, p[0], p[1], 16, 'MAD')
#         p = (p[0] + 8, p[1] + 8)
#         image = cv2.arrowedLine(image, p, best_point, (0, 255, 0), 1)
#
#     cv2.imshow('press space', image)
#     cv2.waitKey(100000)
#     suc, frame1 = vid.read()

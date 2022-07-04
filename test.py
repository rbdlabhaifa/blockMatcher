import cv2

from python_block_matching.utils import *
from python_block_matching.cost_functions import *
from python_block_matching.algorithms import *



frame = cv2.imread('frame10.png')
co = frame.copy()
frame = grayscale(frame)

ras_blocks = set()
our_blocks = set()
with open('frame10.txt') as f:
    for line in f:
        ras_blocks |= {eval(line)}
min_d = (0, 100)
b_blocks = []
for t in range(1000, 10000, 100):
    our_blocks = set()
    for x, y, w, h in intra_frame_mb_partition(frame, t):

        our_blocks |= {(x - w//2, y  - h//2, w, h)}
    difference = 100 * (len(ras_blocks.difference(our_blocks)) / len(ras_blocks))
    print(difference)
    if difference < min_d[1]:
        min_d = (t, difference)
        b_blocks = our_blocks
print('min diff = ', min_d)
for x, y, w, h in b_blocks:
    print(x, y)
    co = cv2.rectangle(co, (x, y), (x + w, y + h), color=(0, 255, 0))
# for x, y, w, h in ras_blocks.difference(b_blocks):
#     co = cv2.rectangle(co, (x, y), (x + w, y + h), color=(100, 255, 0))
cv2.imshow('afafa', co)
cv2.waitKey()

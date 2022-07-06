import cv2

from python_block_matching.utils import *
from python_block_matching.cost_functions import *
from python_block_matching.algorithms import *

vid = cv2.VideoCapture('rawVideo.h264')
for i in range(10):
    a, frame = vid.read()
a, frame = vid.read()
co = frame.copy()
# frame = grayscale(frame)

ras_blocks = set()
our_blocks = set()
with open('frame10.txt') as f:
    for line in f:
        ras_blocks |= {eval(line)}
min_d = (0, 100)
b_blocks = []
for t in range(30500, 30501, 100):
    our_blocks = set()
    for x, y, w, h in intra_frame_mb_partition(frame, t):
        our_blocks |= {(x + w // 2, y + h // 2, w, h)}
    difference = 100 * (len(ras_blocks.difference(our_blocks)) / len(ras_blocks))
    print(t, difference)
    if difference < min_d[1]:
        min_d = (t, difference)
        b_blocks = our_blocks
print('min diff = ', min_d)

for x, y, w, h in b_blocks:
    co = cv2.rectangle(co, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color=(0, 255, 0))
for x, y in get_macro_blocks(frame, 16):
    x += 8
    y += 8
    if {(x, y, 16, 16), (x, y, 16, 8), (x, y, 8, 16), (x, y, 8, 8)}.isdisjoint(ras_blocks):
        if (x, y, 16, 16) in b_blocks:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        co = cv2.rectangle(co, (x - 8, y - 8), (x + 8, y + 8), color=color)
for x, y, w, h in ras_blocks:
    if (x, y, w, h) in b_blocks:
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)
    co = cv2.rectangle(co, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color=color)


print('GREEN=our blocks, BLUE=blocks in both, RED=only RP')
cv2.imshow('afafa', co)
cv2.waitKey()

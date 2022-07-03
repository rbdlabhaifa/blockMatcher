from python_block_matching.utils import *
from python_block_matching.cost_functions import *
from python_block_matching.algorithms import *



frame = cv2.imread('frame1.png')
co = frame.copy()
frame = grayscale(frame)
for x, y, w, h in intra_frame_mb_partition(frame, threshold=1200):
    co = cv2.rectangle(co, (x, y), (x + w, y + h), color=(0, 0, 255))

cv2.imshow('bla bla bla', co)
cv2.waitKey()
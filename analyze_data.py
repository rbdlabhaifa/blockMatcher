import numpy as np
import cv2
from python_block_matching import  algorithms, cost_functions, utils
# Open And Parse Data

file = open("almost360.csv", "r")
cap = cv2.VideoCapture("allmost360.h264")
data = file.read()
data = data.splitlines()

frame_data = np.array([])

for line in data:
    # DATA: framenum,source,blockw,blockh,srcx,srcy,dstx,dsty,flags
    line = line.replace(" ", "")
    line = line.split(",")
    for val in line:
        val = int(val)
    frame_data = np.append(frame_data, np.array(line))

file.close()

# Analyze Data

output = open("output_analasys.csv", "a")
ret, vid_frame = cap.read()

assert ret

for frame in frame_data:
    ret, vid_frame = cap.read()
    sse_general = {"vertical": [], "horizontal": []}
    sse_cut_only = {"vertical": [], "horizontal": []}

    full_count = 0
    vertical_cut_count = 0
    horizontal_cut_count = 0
    quarter_cut_count = 0

    for mb in frame:
        if mb[2] == 16 and mb[3] == 16:
            # Full macro block
            full_count += 1

        if mb[2] == 16 and mb[3] == 8:
            # Horizontal Cut
            horizontal_cut_count += 1

        if mb[2] == 8 and mb[3] == 16:
            # Vertical Cut
            vertical_cut_count += 1

        if mb[2] == 8 and mb[3] == 8:
            # Quarter Cut
            quarter_cut_count += 1

import os
import cv2


path = 'C:/Users/BenGo/PycharmProjects/blockMatcher/Dictionary/data/optitrack/2/'
out = 'tmp/'

files = list(sorted(os.listdir(path[:-1]), key=lambda x: int(x.replace('.jpg', ''))))
base_frame = cv2.imread(path + files[0])
frame_num = 0
for file in range(1, len(files)):
    cv2.imwrite(f'{out}{frame_num}.jpg', base_frame)
    frame_num += 1
    cv2.imwrite(f'{out}{frame_num}.jpg', cv2.imread(path + files[file]))
    frame_num += 1

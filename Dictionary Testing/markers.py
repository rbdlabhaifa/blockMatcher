import os
import cv2
import numpy as np
from python_block_matching import *
from mv_dictionary import MVMapping


folder_path = '../markers.cpp files/data1'
save_to = '/home/rani/PycharmProjects/blockMatcher/Dictionary Testing/trained dicts/data1 dict'
read_from = None


# read frame images
jpg_files = []
for file in os.listdir(folder_path):
    if file.endswith('.jpg') or file.endswith('.png'):
        jpg_files.append(file)
jpg_files = sorted(jpg_files, key=lambda x: int(x.replace('frame', '')[:-4]))

# read .csv files
rotations = []
with open(f'{folder_path}/rot_rigid_drone0.csv', 'r') as csv_file:
    for rotation in csv_file:
        rotations.append(float(rotation))

# calculate motion vectors.
motion_vectors = []
for i in range(1, len(jpg_files)):
    ref_frame = cv2.imread(f'{folder_path}/{jpg_files[i - 1]}')
    ref_frame = np.flipud(ref_frame)
    ref_frame = ref_frame[0:-160, :]
    cur_frame = cv2.imread(f'{folder_path}/{jpg_files[i]}')
    cur_frame = np.flipud(cur_frame)
    cur_frame = cur_frame[0:-160, :]
    motion_vectors.append(BlockMatching.get_motion_vectors(cur_frame, ref_frame))

# add to a dictionary and save to a file.
mv_dictionary = MVMapping(read_from)
for i in range(1, len(rotations)):
    rot = abs(rotations[i] - rotations[i - 1])
    mv_dictionary[motion_vectors[i - 1]] = rot
    print(f'frame {i}, rotation = {rot}')
mv_dictionary.save_to(save_to)

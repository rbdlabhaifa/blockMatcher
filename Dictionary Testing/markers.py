import os
import cv2
from python_block_matching import *
from mv_dictionary import MVMapping


folder_path = ''
save_file = ''


# read jpg and csv files
all_files = os.listdir(folder_path)
jpg_files = list(filter(lambda x: x.endswith('.jpg'), all_files))
csv_files = list(filter(lambda x: x.endswith('.csv'), all_files))


# sort lists by frame index
jpg_files = sorted(jpg_files, key=lambda x: int(x.replace('frame', '')[:-4]))
csv_files = sorted(csv_files, key=lambda x: int(x.replace('rot_rigid_drone', '')[:-4]))


# calculate motion vectors.
motion_vectors = []
for i in range(1, len(jpg_files)):
    ref_frame = cv2.imread(jpg_files[i - 1])
    cur_frame = cv2.imread(jpg_files[i])
    motion_vectors.append(BlockMatching.get_motion_vectors(cur_frame, ref_frame))


# get rotation.
rotations = []
for i in range(1, len(csv_files)):
    pass


# add to a dictionary and save to a file.
mv_dictionary = MVMapping()
for vectors, rotation in zip(motion_vectors, rotations):
    mv_dictionary[vectors] = rotation
mv_dictionary.save_to(save_file)

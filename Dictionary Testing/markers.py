import os
import cv2
from python_block_matching import *
from mv_dictionary import MVMapping


folder_path = ''
save_file = ''


# read frame and rotation files
jpg_files = []
csv_files = []
for file in os.listdir(folder_path):
    if file.endswith('.jpg'):
        jpg_files.append(file)
    elif file.endswith('.csv'):
        csv_files.append(file)


# sort lists
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

print(rotations)


# add to a dictionary and save to a file.
# mv_dictionary = MVMapping()
# for vectors, rotation in zip(motion_vectors, rotations):
#     mv_dictionary[vectors] = rotation
# mv_dictionary.save_to(save_file)

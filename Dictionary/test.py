import os
import numpy as np
from python_block_matching import *
from dictionary import MVMapping
import cv2
from Dictionary.images.synthetic3D import generate_pictures_2_angles


def calculate_rotation(mv):
    fov_x = 60
    res_x = 480
    vector_length = abs(mv[2] - mv[0])
    conv_value = (np.tan(np.deg2rad(fov_x / 2)) / (res_x / 2))
    vector_length *= conv_value
    mv = list(mv)
    mv[0] -= res_x / 2
    mv[2] -= res_x / 2
    a = np.sqrt((mv[0] * conv_value) ** 2 + 1)
    b = np.sqrt((mv[2] * conv_value) ** 2 + 1)
    c = vector_length
    return np.rad2deg(np.arccos((-(c ** 2) + a ** 2 + b ** 2) / (2 * a * b)))


def try_ego_rotation_dicts():
    im_shape = [360, 360]
    square_size = 32
    mv_dict = MVMapping('saved dictionaries/chess_plane_ego_rot_0-8_steps.pickle')
    while True:
        angle, step = eval(input('enter angle and step').replace(' ', ','))
        chess = np.zeros((im_shape[0] * 4, int(im_shape[1] * 4), 3), dtype=np.uint8)
        for i in range(int(chess.shape[0] / square_size)):
            for j in range(int(chess.shape[1] / square_size)):
                chess[square_size * j:square_size * (j + 1), square_size * i:square_size * (i + 1)] = np.random.randint(
                    0,
                    256,
                    3,
                    np.uint8)
        f1, f2 = generate_pictures_2_angles(chess, angle, angle + step, im_shape)
        print('dictionary found: ', mv_dict[BlockMatching.get_motion_vectors(f2, f1)])


def view_motion_vectors_from_frames(folder_path: str):
    # read frame images
    jpg_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.jpg') or file.endswith('.png'):
            jpg_files.append(file)
    jpg_files = sorted(jpg_files, key=lambda x: int(x.replace('frame', '')[:-4]))
    mvs = []
    for i in range(1, len(jpg_files)):
        ref_frame = cv2.imread(f'{folder_path}/{jpg_files[i - 1]}')
        cur_frame = cv2.imread(f'{folder_path}/{jpg_files[i]}')
        mvs.append(BlockMatching.get_motion_vectors(cur_frame, ref_frame))

    for i in range(1, len(jpg_files)):
        ref_frame = cv2.imread(f'{folder_path}/{jpg_files[i - 1]}')
        cur_frame = cv2.imread(f'{folder_path}/{jpg_files[i]}')
        ref_frame = BMFrame(ref_frame)
        ref_frame.draw_motion_vector(mvs[i], (0, 0, 255), 1)
        ref_frame.show()


def compare_dict_with_vid(frame_folder_path: str, csv_path: str, dict: MVMapping):
    # read frame images
    jpg_files = []
    for file in os.listdir(frame_folder_path):
        if file.endswith('.jpg') or file.endswith('.png'):
            jpg_files.append(file)
    jpg_files = sorted(jpg_files, key=lambda x: int(x.replace('frame', '')[:-4]))

    rotations = []
    with open(csv_path, 'r') as csv_file:
        for rotation in csv_file:
            rotations.append(float(rotation))
    new_rot = []
    for i in range(1, len(rotations)):
        new_rot.append(round(abs(rotations[i] - rotations[i - 1]), 2))
    rotations = new_rot
    dict_rot = []
    for i in range(1, len(jpg_files)):
        ref_frame = cv2.imread(f'{frame_folder_path}/{jpg_files[i - 1]}')
        ref_frame = np.flipud(ref_frame)
        ref_frame = ref_frame[0:-160, :]
        cur_frame = cv2.imread(f'{frame_folder_path}/{jpg_files[i]}')
        cur_frame = np.flipud(cur_frame)
        cur_frame = cur_frame[0:-160, :]
        mbs = BlockMatching.get_macro_blocks(cur_frame)
        mvs = BlockMatching.get_motion_vectors(cur_frame, ref_frame, current_frame_mb=mbs)
        cf = BlockMatching.form_compensated_frame(ref_frame, mbs, mvs)
        cv2.imshow('', cf)
        cv2.waitKey()
        dict_rot.append(dict[mvs])
    return np.abs(np.subtract(rotations, dict_rot)/rotations).sum() / len(dict_rot)


if __name__ == '__main__':
    a = compare_dict_with_vid('optitrack/data2',
                              'optitrack/data2/rot_rigid_drone0.csv', MVMapping(
            'saved dictionaries/data1'))
    print(a)

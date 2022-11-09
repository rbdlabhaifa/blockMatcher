import os

import cv2

from formula import Formula
from block_matching import BlockMatching
import numpy as np
import matplotlib.pyplot as plt


def rename(path):
    p = path
    m = min([int(x[:-4]) for x in os.listdir(p)])
    for i in sorted(os.listdir(p), key=lambda x: int(x[:-4])):
        os.rename(p + '/' + i, p + '/' + str(int(i[:-4]) - m) + '.png')


def formula():

    path_to_data = '/media/rani/BENJOBI/runs/3'

    tello_frames = []
    tello_times = []
    for i in sorted(set(os.listdir(path_to_data)) ^ {'rotation.csv', 'translation.csv'}, key=lambda x: int(x[:-4])):
        tello_frames.append(f'{path_to_data}/{i}')
        tello_times.append(int(i[:-4]))

    """
    opti_angles = []
    opti_times = []
    with open(f'{path_to_data}/rotation.csv') as file:
        file.readline()
        for i in file:
            pitch, yaw, roll, time = i.split()
            opti_angles.append(round(abs(float(roll)), 2))
            opti_times.append(int(time))

    angles = []
    sum_of_angles = 0
    start_from = 0
    for tello_time in tello_times:
        for i, opti_time in enumerate(opti_times):
            if i <= start_from:
                continue
            sum_of_angles += opti_angles[i]
            if tello_time < opti_time:
                angles.append(sum_of_angles)
                sum_of_angles = 0
                start_from = i
                break
    """
    camera_matrix = np.array([
        [907.15425425, 0, 478.97775485],
        [0, 907.55348533, 346.90456115],
        [0, 0, 1]
    ])
    frame_vectors = BlockMatching.get_ffmpeg_motion_vectors(tello_frames, save_to=f'{path_to_data}/vid.h264', on_raspi=True, repeat_first_frame=False)
    frame_angle_diff = []
    for i, vectors in enumerate(frame_vectors):
        formula_solutions = Formula.calculate(vectors, camera_matrix, 'y', decimal_places=2, interval=(-10, 10))
        if len(formula_solutions) == 0:
            frame_angle_diff.append(None)
            continue
        formula_solutions = abs(max(formula_solutions.items(), key=lambda x: x[1])[0])
        angle_diff = angles[i + 1] - formula_solutions
        frame_angle_diff.append(abs(angle_diff))

    print(frame_angle_diff)

    x_axis, y_axis = [], []
    for i in range(len(frame_angle_diff)):
        if frame_angle_diff[i] is not None:
            y_axis.append(frame_angle_diff[i])
            x_axis.append(i + 2)

    plt.scatter(x_axis, y_axis, c='blue')
    plt.title('Difference of OptiTrack angle and motion-vectors angle per frame')
    plt.xlabel('Frame Number')
    plt.ylabel('Angle difference')
    plt.show()


if __name__ == '__main__':
    # rename('/home/rani/PycharmProjects/blockMatcher/data/drone/8')
    formula()

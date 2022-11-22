import os

import cv2

from formula import Formula
from block_matching import BlockMatching
import numpy as np
import matplotlib.pyplot as plt


def rename(path):
    p = path
    listdir = [x for x in os.listdir(p) if x.endswith('.png')]
    sortedlist = list(sorted(listdir, key=lambda x: int(x[:-4])))
    for idx, i in enumerate(sortedlist):
        os.rename(p + '/' + i, p + '/' + str(idx) + '.png')


def formula(path_to_data, raspi, repeating, interval):
    tello_frames = []
    tello_times = []
    for i in sorted(set(os.listdir(path_to_data)).difference({'.~lock.rotation.csv#', 'vid.h264', 'rotation.csv', 'translation.csv'}),
                    key=lambda x: int(x[:-4]) ):
        tello_frames.append(f'{path_to_data}/{i}')
        tello_times.append(int(i[:-4]))

    opti_angles = []
    opti_times = []
    with open(f'{path_to_data}/rotation.csv') as file:
        file.readline()
        for i in file:
            pitch, yaw, roll, time = i.split()
            opti_angles.append(round(float(roll), 2))
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

    camera_matrix = np.array([
        [907.15425425, 0, 478.97775485],
        [0, 907.55348533, 346.90456115],
        [0, 0, 1]
    ])
    if not raspi:
        frame_vectors = BlockMatching.get_ffmpeg_motion_vectors(tello_frames, save_to='as_vid.mp4', repeat_first_frame=repeating, first_frame_interval=interval)
    else:
        frame_vectors = BlockMatching.extract_motion_data(f'{path_to_data}/vid.h264')
        # frame_vectors = BlockMatching.get_ffmpeg_motion_vectors(tello_frames, save_to=f'{path_to_data}/vid.h264',
        #                                                         on_raspi=True, repeat_first_frame=False)

    frame_angle_diff = []
    formula_angles = []
    idk = 0
    for i, vectors in enumerate(frame_vectors):
        if repeating and i % 2 != 0:
            continue
        formula_solutions = Formula.calculate(vectors, camera_matrix, 'y', decimal_places=2, interval=(-10, 10))
        if len(formula_solutions) == 0:
            frame_angle_diff.append(None)
            continue
        formula_solutions = max(formula_solutions.items(), key=lambda x: x[1])[0]
        formula_angles.append(formula_solutions)
        if repeating:
            idk += angles[i // 2 + 1]
            angle_diff = 100 * (abs((idk - formula_solutions) / idk))
            if (i // 2) % interval == 0:
                idk = 0
        else:
            angle_diff = 100 * (abs((angles[i + 1] - formula_solutions) / angles[i + 1]))

        frame_angle_diff.append(angle_diff)
        # f = f'{path_to_data}/{tello_times[i // 2]}.png'
        # f = BlockMatching.draw_motion_vectors(cv2.imread(f), vectors)
        # cv2.imshow('', f)
        # cv2.waitKey(1)

    x_axis, y_axis = [], []
    for i in range(len(frame_angle_diff)):
        if frame_angle_diff[i] is not None:
            y_axis.append(frame_angle_diff[i])
            x_axis.append(i + 2)

    y_axis = np.array(y_axis)

    y_axis2 = []
    x_axis2 = []

    all_frame_count = 0
    for i in np.arange(1, 101, 1):
        frame_count = np.where((y_axis < i) & (y_axis >= (i - 1)))[0].shape[0]
        all_frame_count += frame_count
        if frame_count is None or not isinstance(frame_count, int):
            y_axis2.append(0)
        else:
            y_axis2.append(frame_count)
        x_axis2.append(i)

    return x_axis2, y_axis2


def graph(x_axis, y_pc, raspi, compare, title='Angle error to the number of frames with that error'):
    if raspi:
        lbl = "Raspberry Pi 0"
        lbl_compare = "Computer"
    else:
        lbl = "Computer"
        lbl_compare = "Raspberry Pi 0"
    if compare:
        xa_rasp, ya_rasp = formula('/media/txp1/BENJOBI/runs/1',True , False, 0)
        plt.plot(x_axis, ya_rasp, c='orange', label=lbl_compare)
    plt.plot(x_axis, y_pc, c='blue', label=lbl)
    plt.title(title)
    plt.xlabel('error in %')
    plt.ylabel('# of frames')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    raspi = True
    compare = False
    xa_comp, ya_comp = formula('/media/txp1/BENJOBI/runs/1', raspi, True, 10)
    graph(xa_comp, ya_comp, raspi, compare)

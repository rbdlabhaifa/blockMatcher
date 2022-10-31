import os
import threading
import time
from formula import Formula
from mvextractor.videocap import VideoCap
import cv2
from block_matching import BlockMatching
import numpy as np
from djitellopy import Tello


def rename(path):
    p = path
    m = min([int(x[:-4]) for x in os.listdir(p)])
    for i in sorted(os.listdir(p), key=lambda x: int(x[:-4])):
        os.rename(p + '/' + i, p + '/' + str(int(i[:-4]) - m) + '.png')


def form():
    camera_matrix = np.array([
        [907.15425425, 0, 478.97775485],
        [0, 907.55348533, 346.90456115],
        [0, 0, 1]
    ])
    p = '/home/rani/PycharmProjects/blockMatcher/data/optitrack/7.mp4'
    angles = []
    with open(p.replace('.mp4', '.csv'), 'r') as f:
        for i in f:
            pitch, yaw, roll = eval(i.replace(' ', ','))
            angles.append(pitch)
    delta_angles = []
    for i in range(len(angles) - 1):
        delta_angles.append(abs(angles[i + 1] - angles[i]))
    vectors_cap = VideoCap()
    vectors_cap.open(p)
    was_read, frame, vectors, frame_type, _ = vectors_cap.read()
    i = 0
    iframes = 0
    while was_read:
        if frame_type == 'I':
            print('i frame')
            iframes += 1
            was_read, frame, vectors, frame_type, _ = vectors_cap.read()
            continue
        mvs = vectors[:, 3:7]
        sols = Formula.calculate(mvs, camera_matrix, 'y', decimal_places=2, interval=(-10, 10), remove_zeros=True)
        was_read, frame, vectors, frame_type, _ = vectors_cap.read()
        frame = BlockMatching.draw_motion_vectors(frame, mvs)
        cv2.imshow('', frame)
        cv2.waitKey()
        if len(sols):
            sol = max(sols.items(), key=lambda x: x[1])[0]
            print('i:', i, 'real angle:', delta_angles[i], 'formula solution:', sol, 'error:', delta_angles[i] - sol)
        i += 1
    print(f'{iframes=}')

    # import matplotlib.pyplot as plt
    #
    # Year = [i for i in all_sols.keys()]
    # Unemployment_Rate = [i for i in all_sols.values()]
    #
    # plt.plot(Year, Unemployment_Rate)
    # # plt.title('Unemployment Rate Vs Year')
    # # plt.xlabel('sum of angles')
    # # plt.ylabel('Unemployment Rate')
    # plt.show()


if __name__ == '__main__':
    # rename('/home/rani/PycharmProjects/blockMatcher/data/drone/8')
    form()

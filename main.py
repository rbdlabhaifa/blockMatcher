import cv2
import numpy as np
import os
from block_matching import BlockMatching
#from formula import *


def view_vectors(path_to_data, suffix='.png'):
    if suffix == '.jpg':
        with open(path_to_data + '.csv', 'r') as file:
            rots = []
            for i in file:
                rots.append(float(i))
        angles = [abs(rots[i] - rots[i - 1]) for i in range(1, len(rots))]
        motion_vectors = BlockMatching.extract_motion_data(path_to_data + '.mp4')
    else:
        frames = [f'{path_to_data}/{i}' for i in
                  sorted(os.listdir(path_to_data), key=lambda x: int(x.replace(suffix, '')))]
        motion_vectors = BlockMatching.get_ffmpeg_motion_vectors(frames)
    for i, vectors in enumerate(motion_vectors):
        if suffix == '.png':
            if i % 2 == 1:
                continue
            print('angle=', 0.1 * ((i // 2) + 1))
        else:
            print('angle=', angles[i])
        base_image = cv2.imread(path_to_data + '/0' + suffix)
        base_image = BlockMatching.draw_motion_vectors(base_image, vectors)
        cv2.imshow('', base_image)
        input('enter something')


if __name__ == '__main__':
    # fov_x = 60
    # fov_y = 60
    # width, height = 1000, 1000
    # fx = width / (2 * np.tan(np.deg2rad(fov_x)))
    # fy = height / (2 * np.tan(np.deg2rad(fov_y)))
    # cx, cy = width // 2, height // 2
    # axis = 'y'
    # mat = np.array([
    #     [fx, 0, cx],
    #     [0, fy, cy],
    #     [0, 0, 1]
    # ])
    #
    # save_to = f'/home/rani/Desktop/graphs/computer/webcam/optitrack'
    # p = f'/home/rani/PycharmProjects/blockMatcher/data/optitrack/8'
    # mat = np.array(
    #     [[646.74302145  , 0. ,        341.39908641],
    #      [0.        , 649.06238853, 207.9928129],
    #     [0., 0., 1.]]
    #
    #
    # )
    import os
    #Formula.run_on_data(p, mat, 'x', '%f', p + '.csv', False, show=True)
    i = f'/home/ben/PycharmProjects/blockMatcher/data/synthetic/rotation - x'
    frames = [f'{i}/{j}' for j in sorted(os.listdir(i), key=lambda x: int(x[:-4]))]
    BlockMatching.get_ffmpeg_motion_vectors(frames, i  +'.h264', on_raspi=True)
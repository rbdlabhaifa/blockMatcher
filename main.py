# ===================================================== IMPORTS ====================================================== #


# from formula import Formula
import cv2
import numpy as np
from typing import Any, Dict
# import sympy
# from sympy import Symbol, simplify, diff, solve, Float, sin, cos, init_printing, pprint, lambdify
# import mpmath
import pickle
import os
from typing import Tuple
from block_matching import BlockMatching
from formula import Formula


# ===================================================== FORMULA ====================================================== #


def check_formula_on_synthetic_data(path_to_data: str, camera_matrix, axis, title: str = '', save_to: str = None,
                                    debug: bool = False, from_video: bool = False):
    if from_video:
        motion_vectors = BlockMatching.extract_motion_data(path_to_data)
    else:
        frames = [f'{path_to_data}/{i}' for i in sorted(os.listdir(path_to_data), key=lambda x: int(x.replace('.png', '')))][5:]
        motion_vectors = BlockMatching.get_ffmpeg_motion_vectors_with_cache(frames)
    angle = []
    for i, vectors in enumerate(motion_vectors):
        if i % 2 == 1:
            continue
        if debug:
            base_image = cv2.imread(path_to_data + '/0.png')
            base_image = BlockMatching.draw_motion_vectors(base_image, vectors)
            cv2.imshow('', base_image)
            cv2.waitKey()
        ans = Formula.calculate(vectors, camera_matrix, axis, interval=(-50, 50), remove_zeros=True)
        Formula.graph_solutions(ans, title, save_to=save_to, show=debug)
    return angle


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
        motion_vectors = BlockMatching.get_ffmpeg_motion_vectors_with_cache(frames)
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


# ===================================================== MAIN ========================================================= #


if __name__ == '__main__':
    p = ''
    fov_x = 60
    fov_y = 60
    width, height = 1000, 1000
    fx = width / (2 * np.tan(np.deg2rad(fov_x)))
    fy = height / (2 * np.tan(np.deg2rad(fov_y)))
    cx, cy = width / 2, height / 2
    mat = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    mat = np.array(
        [
            [629.94662448, 0, 316.23232917],
            [0, 629.60772237, 257.64459816],
            [0, 0, 1]
        ]
    )
    p = '/home/rani/PycharmProjects/blockMatcher/data/webcam/2'
    results = check_formula_on_synthetic_data(p, mat, 'y', debug=True)

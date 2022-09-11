import cv2
import numpy as np
from typing import Any
from sympy import Symbol, simplify, diff, solve, Float, sin, cos, init_printing
import mpmath
import pickle
import os
import subprocess
from block_matching import BlockMatching
from Dictionary.dictionary import MVMapping


# ====================================================== UTILS ====================================================== #


def load_expression(expression_path: str):
    with open(expression_path) as f:
        expression = pickle.load(f)
    return expression


def calculate_expression(cam_mat, save_to: str = None):
    init_printing(use_unicode=True, wrap_line=False, no_global=False)
    inv_cam_mat = np.linalg.inv(cam_mat)

    after_image_x = Symbol('after_x')
    before_image_x = Symbol('before_x')
    after_image_y = Symbol('after_y')
    before_image_y = Symbol('before_y')
    Fx = Symbol('Fx')
    Fy = Symbol('Fy')
    Cx = Symbol('Cx')
    Cy = Symbol('Cy')
    inv_Fx = Symbol('inv_Fx')
    inv_Fy = Symbol('inv_Fy')
    inv_Cx = Symbol('inv_Cx')
    inv_Cy = Symbol('inv_Cy')
    alpha = Symbol('alpha')

    expression_a = (((Cx * sin(alpha) * (inv_Fy * before_image_y + inv_Cy) + Cx * cos(alpha) + Fx * (
            inv_Fx * before_image_x + inv_Cx)) /
                     (cos(alpha) + sin(alpha) * (inv_Fy * before_image_y + inv_Cy))) - after_image_x) ** 2
    expression_b = ((((-Fy) * sin(alpha) + (inv_Fy * before_image_y + inv_Cy) * (
            Fy * cos(alpha) + Cy * sin(alpha)) + Cy * cos(alpha)) /
                     (cos(alpha) + sin(alpha) * (inv_Fy * before_image_y + inv_Cy))) - after_image_y) ** 2

    values = [(Cx, cam_mat[0][2]), (Cy, cam_mat[1][2]), (Fx, cam_mat[0][0]), (Fy, cam_mat[1][1]),
              (inv_Fx, inv_cam_mat[0][0]), (inv_Fy, inv_cam_mat[1][1]),
              (inv_Cx, inv_cam_mat[0][2]), (inv_Cy, inv_cam_mat[1][2])]

    expression_a.subs(values)
    expression_b.subs(values)

    dif_a = diff(expression_a, alpha)
    dif_b = diff(expression_b, alpha)

    simple_a = simplify(dif_a)
    simple_b = simplify(dif_b)

    solve_a = solve(simple_a, alpha)
    solve_b = solve(simple_b, alpha)

    s11 = solve_a[0].subs(values)
    s12 = solve_a[1].subs(values)
    s13 = solve_a[2].subs(values)
    s14 = solve_a[3].subs(values)
    s21 = solve_b[0].subs(values)
    s22 = solve_b[1].subs(values)

    s11 = simplify(s11)
    s12 = simplify(s12)
    s13 = simplify(s13)
    s14 = simplify(s14)
    s21 = simplify(s21)
    s22 = simplify(s22)

    s11 = mpmath.degrees(s11)
    s12 = mpmath.degrees(s12)
    s13 = mpmath.degrees(s13)
    s14 = mpmath.degrees(s14)
    s21 = mpmath.degrees(s21)
    s22 = mpmath.degrees(s22)

    ret = np.array([s11, s12, s13, s14, s21, s22])

    if save_to is not None:
        with open(save_to, 'w') as f:
            pickle.dump([s11, s12, s13, s14, s21, s22], f)

    return ret


def calculate_angle(expressions: Any, vector: tuple):
    after_image_p = vector[2], vector[3]
    before_image_p = vector[0], vector[1]
    init_printing(use_unicode=True, wrap_line=False, no_global=False)
    ret = []
    after_image_x = Symbol('after_x')
    before_image_x = Symbol('before_x')
    after_image_y = Symbol('after_y')
    before_image_y = Symbol('before_y')
    values = [(after_image_x, after_image_p[0]),
              (after_image_y, after_image_p[1]), (before_image_x, before_image_p[0]),
              (before_image_y, before_image_p[1])]
    for idx, i in enumerate(expressions):
        temp = i.subs(values)
        temp_sol = simplify(temp)
        if type(i) is not Float or i < 0.:
            ret.append(temp_sol)
    return ret


def create_rotation_video(image_folder: str, temporary_folder: str):
    files = list(sorted(os.listdir(image_folder), key=lambda x: int(x.replace('.png', '').replace('.jpg', ''))))
    base_frame = cv2.imread(f'{image_folder}/{files[0]}')
    frame_number = 0
    for i in range(1, len(files)):
        cv2.imwrite(f'{temporary_folder}/{frame_number}.png', base_frame)
        frame_number += 1
        cv2.imwrite(f'{temporary_folder}/{frame_number}.png', cv2.imread(f'{image_folder}/{i}.png'))
        frame_number += 1
    subprocess.run(['ffmpeg', '-i', f'\"{temporary_folder}/%d.png\"',
                    '-c:v', 'h264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', f'{image_folder}/out.mp4'])


# ====================================================== TESTS ====================================================== #


def motion_vectors():
    pass


# ====================================================== MAIN ======================================================= #


if __name__ == '__main__':
    path = '/home/rani/PycharmProjects/blockMatcher/Dictionary/'
    rim = cv2.imread(path + 'data/optitrack/1/0.jpg')
    cim = cv2.imread(path + 'data/optitrack/1/9.jpg')
    mvs = BlockMatching.get_opencv_motion_vectors(rim, cim)
    print(mvs)
    im = BlockMatching.draw_motion_vectors(rim, mvs)
    cv2.imshow('', im)
    cv2.imshow('1', cim)
    cv2.waitKey()
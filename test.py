from typing import Tuple

import cv2
import numpy as np
from block_matching import BlockMatching
from typing import Any
from sympy import *
import numpy as np
import mpmath


# ====================================================== UTILS ====================================================== #


def calculate_expression(cam_mat):
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

    s1 = expression_a.subs(values)
    s2 = expression_b.subs(values)

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
    return ret


def calculate_angle(expressions:  Any, vector: tuple):
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
    for idx,i in enumerate(expressions):
        temp = i.subs(values)
        temp_sol = simplify(temp)
        if type(i) is not Float or i < 0.:
            ret.append(temp_sol)
    print(ret)
    return ret


# ====================================================== TESTS ====================================================== #


def test_motion_vectors():
    """
    Tests how many motion vectors properly represent the rotation that occurred.

    :param output: The path to the output file.
    """
    EXTRACT_PATH = '/home/rani/PycharmProjects/blockMatcher/Extra Code/extract motion data/motionVectors'
    VIDEO_PATH = '/home/rani/PycharmProjects/blockMatcher/Dictionary/data/gradient/2.mp4'
    VIDEO_FOV = [60, 40]
    was_read, frame = cv2.VideoCapture(VIDEO_PATH).read()
    if not was_read:
        raise ValueError('Could not read frame.')
    output = ''
    width, height = frame.shape[1], frame.shape[0]
    camera_matrix = np.array([
        [width / (2 * np.tan(np.deg2rad(VIDEO_FOV[0]))), 0, width / 2],
        [0, height / (2 * np.tan(np.deg2rad(VIDEO_FOV[1]))), height / 2],
        [0, 0, 1]
    ])
    expressions = calculate_expression(camera_matrix)
    motion_data = BlockMatching.extract_motion_data(EXTRACT_PATH, VIDEO_PATH)
    for i in range(len(motion_data)):
        if i % 2 != 0:
           continue
        for vector in motion_data[i]:
            print(f'real angle: {0.1 * (i / 2 + 1)} got: ', calculate_angle(expressions, vector))

# ====================================================== MAIN ======================================================= #


if __name__ == '__main__':
    # The path to the output of the function, None if you want to print.
    OUTPUT = ''
    # Run a test.
    test_motion_vectors()

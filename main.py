# ===================================================== IMPORTS ====================================================== #


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


# ===================================================== FORMULA ====================================================== #


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


# ===================================================== TESTS ======================================================== #


def motion_vectors():
    pass


# ===================================================== MAIN ========================================================= #


def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T

def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result



if __name__ == '__main__':
    import io
    path = 'Dictionary/data/gradient/9/'
    frames = [(path + i) for i in list(sorted(os.listdir(path[:-1]), key=lambda x: int(x.replace('.png', ''))))]
    mvs = BlockMatching.get_ffmpeg_motion_vectors_with_cache(frames)
    base = cv2.imread(frames[0])
    for i in mvs:
        basec = base.copy()
        basec = BlockMatching.draw_motion_vectors(basec, i)
        cv2.imshow('', basec)
        cv2.waitKey()

    # cv2.waitKey()

    # from PIL import Image
    #
    # array = get_gradient_3d(3600, 3600, (90, 220, 255), (255, 0, 0), (False, True, False))
    # array = np.asarray(Image.fromarray(np.uint8(array)))
    # array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('im1.png', array)
    # cv2.imshow('', array)
    # cv2.waitKey()

    # image = cv2.imread('im1.png')
    # latitude, longitude = 360, 180
    # lat_step, lon_step = 1, 1
    # center = image.shape[1] // 2, image.shape[0] // 2
    # for lat in np.arange(0, latitude, lat_step):
    #     im = []
    #
    #     length = 0
    #     while length < center[0]:
    #         final_x = center[0] + (length * np.cos(np.deg2rad(lat)))
    #         final_y = center[1] - (length * np.sin(np.deg2rad(lat)))
    #         im.append(image[int(final_y), int(final_x)])
    #         length += 1
    #
    #     im = [im]
    #     cv2.namedWindow('1')
    #     cv2.moveWindow('1', 0, 900)
    #     cv2.imshow('1', np.array(im * 100))
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    #

    # im1 = cv2.imread('Dictionary/data/gradient/3/0.png')
    # im2 = np.rot90(im1)
    # im3 = (im1 + im2) // 3
    # cv2.imshow('', im3)
    # cv2.waitKey()
    #

    # path = 'Dictionary/data/gradient/8'
    # import os
    # ans = set()
    # for i in os.listdir(path):
    #     print(i)
    #     for j in os.listdir(path):
    #         if i == j:
    #             continue
    #         im1, im2 = cv2.imread(path + '/' + i), cv2.imread(path + '/' + j)
    #         ans.add((im1 == im2).all())
    # print(ans)
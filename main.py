# ===================================================== IMPORTS ====================================================== #


import cv2
import numpy as np
from typing import Any

import sympy
from sympy import Symbol, simplify, diff, solve, Float, sin, cos, init_printing, pprint
import mpmath
import pickle
import os
import subprocess
from IPython.display import display, Math
from block_matching import BlockMatching
from Dictionary.dictionary import MVMapping


# ===================================================== FORMULA ====================================================== #


def load_expression(expression_path: str):
    with open(expression_path, 'rb') as f:
        expression = pickle.load(f)
    '''
    init_printing(use_unicode=True, wrap_line=False, no_global=False)
    for i in expression:
        pprint(i)
    '''
    return expression


def calculate_expression(cam_mat, axis, save_to: str = None):
    init_printing(use_unicode=True, wrap_line=False, no_global=False)
    inv_cam_mat = np.linalg.inv(cam_mat)
    check = True
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

    if axis is 'x':
        Rotation_translation_matrix = sympy.Matrix(
            ([1,0,0,0],[0, cos(alpha), -sin(alpha), 0], [0, sin(alpha), cos(alpha), 0], [0, 0, 0, 1]))
    elif axis is 'y':
        Rotation_translation_matrix = sympy.Matrix(
            ([cos(alpha), 0, -sin(alpha), 0], [0, 1, 0, 0], [sin(alpha), 0, cos(alpha), 0], [0, 0, 0, 1]))
    elif axis is 'z':
        Rotation_translation_matrix = sympy.Matrix(
            ([cos(alpha), -sin(alpha), 0, 0], [sin(alpha), cos(alpha), 0, 0],[0,0,1,0], [0, 0, 1, 1]))

    homogeneous_camera_matrix = sympy.Matrix(([Fx,0,Cx,0],[0,Fy,Cy,0],[0,0,1,0]))
    inverse_camera_and_before_point = sympy.Matrix(([inv_Fx*before_image_x+inv_Cx],[inv_Fy*before_image_y + inv_Cy],[1],[1]))
    sol = homogeneous_camera_matrix * Rotation_translation_matrix * inverse_camera_and_before_point
    expression_a = sol[0] / sol[2]
    expression_b = sol[1] / sol[2]

    dif_a = diff(expression_a, alpha)
    dif_b = diff(expression_b, alpha)

    simple_a = simplify(dif_a)
    simple_b = simplify(dif_b)

    solve_a = solve(simple_a, alpha)
    solve_b = solve(simple_b, alpha)


    if axis is 'x':
        s11 = solve_a[0].subs(values)
        s12 = solve_a[1].subs(values)
        s11 = mpmath.degrees(simplify(s11))
        s12 = mpmath.degrees(simplify(s12))
    elif axis is 'y':
        s21 = solve_b[0].subs(values)
        s22 = solve_b[1].subs(values)
        s21 = simplify(s21)
        s22 = simplify(s22)
    elif axis is 'z':
        s11 = solve_a[0].subs(values)
        s12 = solve_a[1].subs(values)
        s21 = solve_b[0].subs(values)
        s22 = solve_b[1].subs(values)

        s11 = mpmath.degrees(simplify(s11))
        s12 = simplify(s12)
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
        try:
            with open(save_to, 'wb') as f:
                pickle.dump([s11, s12, s13, s14, s21, s22], f)
        except (TypeError, Exception):
            pass
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
        if type(temp_sol) is Float and -2 < temp_sol < 2:
            temp_sol = round(temp_sol,1)
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


def shuffle_function(width: int):
    from itertools import combinations
    from random import choice
    what = np.array([i for i in range(width)])
    np.random.shuffle(what)
    def shuffle(arr):
        nonlocal what
        for i in range(arr.shape[0]):
            for j in range(len(what)):
                x = what[j]
                # arr[j, i], arr[x, i] = arr[x, i], arr[j, i]
                arr[i, j], arr[i, x] = arr[i, x], arr[i, j]

    return shuffle


def create_dict(path_to_data, path_to_save, rots=tuple([i / 10 for i in range(1, 51, 1)]), debug = False):
    mapping = MVMapping()
    files = list(sorted(os.listdir(path_to_data), key=lambda x: int(x.replace('.png', ''))))
    frames = [path_to_data + '/' + files[i] for i in range(len(files))]
    for i, mvs in enumerate(BlockMatching.get_ffmpeg_motion_vectors_with_cache(frames)):
        if i % 2 == 1:
            continue
        rot = rots[i // 2]
        mapping[mvs] = rot
        if debug:
            print(f'i={i}, rot={rot}')
            base_frame = cv2.imread(frames[0])
            base_frame = BlockMatching.draw_motion_vectors(base_frame, mvs)
            cv2.imshow('debug', base_frame)
            cv2.waitKey()
    mapping.save_to(path_to_save)
    return mapping

def create_compare_data(path_to_data, rots=tuple([i / 10 for i in range(1, 51, 1)]), debug = False):
    compare_data = {}
    #files = list(sorted(os.listdir(path_to_data), key=lambda x: int(x.replace('.png', ''))))
    #frames = [path_to_data + '/' + i for i in files]
    '''
    size = (1000, 1000)
    fov_x = 60
    fov_y = 60
    cam_mat = np.array([[size[0] / (2 * np.tan(np.deg2rad(fov_x))), 0, size[0] / 2],
                        [0, size[1] / (2 * np.tan(np.deg2rad(fov_y))), size[1] / 2],
                        [0, 0, 1]])
    '''
    cam_mat = [[466.98762407 ,  0. ,        320.89256506],
    [  0.,         467.64693224, 192.73899091],
    [  0. ,          0. ,          1.        ]]
    expression = calculate_expression(cam_mat,'x','expression1')
    #expression = load_expression('expression1')
    cap = cv2.VideoCapture(path_to_data)
    mvs = BlockMatching.extract_motion_data(path_to_data)
    for i in range(66):
        sum = 0
        cnt = 0
        pos = {}
        neg = {}
        for x1, y1, x2, y2 in mvs[i]:
            if x1 == x2 and y1 == y2:
                #  cnt = cnt +1
                continue
            rot = calculate_angle(expression,(x1,y1,x2,y2))

            #print(rot)
            #if len(rot) == 0:
            #continue
            #temp_sum = 0
            for k in rot:
                k = float(str(k))
                if k > 0:
                    pos[k] = pos.get(k, 0) + 1
                elif k < 0:
                    neg[k] = neg.get(k, 0) + 1
                #temp_sum = temp_sum + k
        print('max positive (angle, occurrences):', list(sorted(pos.items(), key=lambda x: x[1]))[-5:])
        print('max negative (angle, occurrences):', list(sorted(neg.items(), key=lambda x: x[1]))[-5:])
        #temp_sum = temp_sum/len(rot)
        #sum += temp_sum
        #sum = sum / (len(mvs) - cnt)
        print('===========================================')
        # compare_data[rot] = mvs
        if debug:
            print(f'i={i}, rot={rot}')
            base_frame = cv2.imread(frames[0])
            base_frame = BlockMatching.draw_motion_vectors(base_frame, mvs)
            cv2.imshow('debug', base_frame)
            cv2.waitKey()
    return compare_data


def view_data(path_to_data, rots=tuple([i / 10 for i in range(1, 16, 1)])):
    files = list(sorted(os.listdir(path_to_data), key=lambda x: int(x.replace('.png', ''))))
    frames = [path_to_data + '/' + files[i] for i in range(len(files))]
    print('press ` to capture an image.')
    for i, mvs in enumerate(BlockMatching.get_ffmpeg_motion_vectors_with_cache(frames)):
        if i % 2 == 1:
            continue
        rot = rots[i // 2]
        print(f'i={i}, rot={rot}')
        base_frame = cv2.imread(frames[0])
        base_frame = BlockMatching.draw_motion_vectors(base_frame, mvs, color=(0,0, 0))
        cv2.imshow('debug', base_frame)
        key = cv2.waitKey()
        if key == ord('`'):
            cv2.imwrite(f'{rot}.png', base_frame)

if __name__ == '__main__':
    create_compare_data('/home/txp1/Downloads/city-quay/blockMatcher-master/Dictionary/data/optitrack/1/video.mp4')


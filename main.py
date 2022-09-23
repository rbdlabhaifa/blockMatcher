# ===================================================== IMPORTS ====================================================== #


import cv2
import numpy as np
from typing import Any
import sympy
from sympy import Symbol, simplify, diff, solve, Float, sin, cos, init_printing, pprint
import mpmath
import pickle
import os
from typing import Tuple
from block_matching import BlockMatching


# ===================================================== FORMULA ====================================================== #


def load_expression(expression_path: str):
    with open(expression_path, 'rb') as f:
        expression = pickle.load(f)
    return expression


def calculate_expression(camera_matrix, axis, save_to: str = None):
    init_printing(use_unicode=True, wrap_line=False, no_global=False)

    after_image_x = Symbol('after_x')
    after_image_y = Symbol('after_y')
    before_image_x = Symbol('before_x')
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

    if axis == 'x':
        rotation_translation_matrix = sympy.Matrix((
            [1, 0, 0, 0],
            [0, cos(alpha), -sin(alpha), 0],
            [0, sin(alpha), cos(alpha), 0],
            [0, 0, 0, 1]
        ))
    elif axis == 'y':
        rotation_translation_matrix = sympy.Matrix((
            [cos(alpha), 0, -sin(alpha), 0],
            [0, 1, 0, 0],
            [sin(alpha), 0, cos(alpha), 0],
            [0, 0, 0, 1]
        ))
    elif axis == 'z':
        rotation_translation_matrix = sympy.Matrix((
            [cos(alpha), -sin(alpha), 0, 0],
            [sin(alpha), cos(alpha), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1]
        ))
    else:
        raise ValueError(f'axis must be x, y or z.')

    homogeneous_camera_matrix = sympy.Matrix((
        [Fx, 0, Cx, 0],
        [0, Fy, Cy, 0],
        [0, 0, 1, 0]
    ))
    inverse_camera_and_before_point = sympy.Matrix((
        [inv_Fx * before_image_x + inv_Cx],
        [inv_Fy * before_image_y + inv_Cy],
        [1],
        [1]
    ))
    sol = homogeneous_camera_matrix * rotation_translation_matrix * inverse_camera_and_before_point

    expression_a = ((sol[0] / sol[2]) - after_image_x) ** 2
    expression_b = ((sol[1] / sol[2]) - after_image_y) ** 2

    dif_a = diff(expression_a, alpha)
    dif_b = diff(expression_b, alpha)

    simple_a = simplify(dif_a)
    simple_b = simplify(dif_b)
    solve_a = solve(simple_a, alpha)
    solve_b = solve(simple_b, alpha)

    inverse_camera_matrix = np.linalg.inv(camera_matrix)

    values = {
        Fx: camera_matrix[0, 0],
        Fy: camera_matrix[1, 1],
        Cx: camera_matrix[0, 2],
        Cy: camera_matrix[1, 2],
        inv_Fx: inverse_camera_matrix[0, 0],
        inv_Fy: inverse_camera_matrix[1, 1],
        inv_Cx: inverse_camera_matrix[0, 2],
        inv_Cy: inverse_camera_matrix[1, 2],
    }

    solutions = []

    for i in solve_a:
        solutions.append(mpmath.degrees(simplify(i.subs(values))))
    for i in solve_b:
        solutions.append(mpmath.degrees(simplify(i.subs(values))))

    if save_to is not None:
        try:
            with open(save_to, 'wb') as f:
                pickle.dump(solutions, f)
        except (TypeError, Exception):
            pass

    return solutions


def calculate_angle(expressions: Any, vector: Tuple):
    # init_printing(use_unicode=True, wrap_line=False, no_global=False)
    solutions = []
    after_image_p = vector[2], vector[3]
    before_image_p = vector[0], vector[1]
    after_image_x = Symbol('after_x')
    before_image_x = Symbol('before_x')
    after_image_y = Symbol('after_y')
    before_image_y = Symbol('before_y')
    values = [
        (after_image_x, after_image_p[0]),
        (after_image_y, after_image_p[1]),
        (before_image_x, before_image_p[0]),
        (before_image_y, before_image_p[1])
    ]
    for expression in expressions:
        solution = simplify(expression.subs(values))
        if isinstance(solution, Float) and -5 < solution < 5:
            solution = round(solution, 1)
            solutions.append(solution)
    return solutions


def check_formula_on_synthetic_data(path_to_data: str, expression, debug: bool = False):
    frames = [f'{path_to_data}/{i}' for i in sorted(os.listdir(path_to_data), key=lambda x: int(x.replace('.png', '')))]
    motion_vectors = BlockMatching.get_ffmpeg_motion_vectors_with_cache(frames)
    for i, vectors in enumerate(motion_vectors):
        if i % 2 == 1:
            continue
        if debug:
            base_image = cv2.imread(path_to_data + '/0.png')
            base_image = BlockMatching.draw_motion_vectors(base_image, vectors)
            cv2.imshow('', base_image)
            cv2.waitKey()
            continue
        positive_solutions = {}
        negative_solutions = {}
        for vector in vectors:
            for solution in calculate_angle(expression, vector):
                if solution > 0:
                    positive_solutions[solution] = positive_solutions.get(solution, 0) + 1
                elif solution < 0:
                    negative_solutions[solution] = negative_solutions.get(solution, 0) + 1
        positive_solutions = list(sorted(positive_solutions.items(), key=lambda x: x[1]))[::-1]
        negative_solutions = list(sorted(negative_solutions.items(), key=lambda x: x[1]))[::-1]
        print(f'i={i}, rot={0.1 * (1 + (i // 2))}')
        print('positive solutions (solution, occurrences):', positive_solutions[:8])
        print('negative solutions (solution, occurrences):', negative_solutions[:8])


def check_formula_on_optitrack_data(path_to_data: str, expression, debug: bool = False):
    # frames = [path_to_data + '/' + i for i in sorted(os.listdir(path_to_data), key=lambda x: 66 - int(x.replace('.jpg', '')))]
    motion_vectors = BlockMatching.extract_motion_data(path_to_data + '.mp4')
    with open(path_to_data + '.csv', 'r') as file:
        rots = []
        for i in file:
            rots.append(float(i))
    angles = [abs(rots[i] - rots[i - 1]) for i in range(1, len(rots))]
    angles_by_vectors = []
    for i, vectors in enumerate(motion_vectors):
        # if i % 2 == 1:
        #     continue
        positive_solutions = {}
        negative_solutions = {}
        for vector in vectors:
            for solution in calculate_angle(expression, vector):
                if solution > 0:
                    positive_solutions[solution] = positive_solutions.get(solution, 0) + 1
                elif solution < 0:
                    negative_solutions[solution] = negative_solutions.get(solution, 0) + 1
        if debug:
            base_image = cv2.imread(path_to_data + f'/0.jpg')
            base_image = BlockMatching.draw_motion_vectors(base_image, vectors)
            cv2.imshow('', base_image)
            cv2.waitKey()
        positive_solutions = list(sorted(positive_solutions.items(), key=lambda x: x[1]))[::-1]
        negative_solutions = list(sorted(negative_solutions.items(), key=lambda x: x[1]))[::-1]
        sum_pos = sum([i[1] for i in positive_solutions])
        sum_neg = sum([i[1] for i in negative_solutions])
        print(f'i={i}, rot={angles[i]}')
        print('positive solutions (solution, occurrences):', positive_solutions[:8])
        print('negative solutions (solution, occurrences):', negative_solutions[:8])
        average_pos = 0
        for j in range(len(positive_solutions)):
            average_pos += positive_solutions[j][0] * (positive_solutions[j][1] / sum_pos)
        print('positive average solution:', average_pos)
        average_neg = 0
        for j in range(len(negative_solutions)):
            average_neg += negative_solutions[j][0] * (negative_solutions[j][1] / sum_neg)
        average_sum = len(negative_solutions) + len(positive_solutions)
        average_positive = len(positive_solutions) / average_sum
        average_negative = len(negative_solutions) / average_sum
        print('negative average solution:', average_neg)
        print('averages:', average_pos * average_positive + average_neg * average_negative)
        angles_by_vectors.append(negative_solutions[0][0])
    with open('results.csv', 'w') as file:
        text = 'real angle, calculated angle, difference\n'
        differences = []
        for i in range(len(angles)):
            difference = round(abs(angles[i] - angles_by_vectors[i]), 4)
            text += f'{round(angles[i], 4)}, {round(angles_by_vectors[i], 4)}, {difference}\n'
            differences.append(difference)
        text += f'Average difference: {round(sum(differences) / len(angles), 4)}\n'
        text += f'Max difference: {round(max(differences), 4)}'
        file.write(text)


# ===================================================== MAIN ========================================================= #


if __name__ == '__main__':
    PATH_TO_DATA = os.getcwd() + '/Dictionary/data/synthetic'
    fov_x, fov_y = np.deg2rad(60), np.deg2rad(60)
    width, height = 1000, 1000
    cx, cy = width / 2, height / 2
    fx, fy = width / (2 * np.tan(fov_x)), height / (2 * np.tan(fov_y))
    # expression = calculate_expression(np.array([
    #     [466.98762407, 0, 320.89256506],
    #     [0, 467.64693224, 192.73899091],
    #     [0, 0, 1]
    # ]), 'y', 'optitrack_expression')
    # expression = load_expression('optitrack_expression')
    # check_formula_on_optitrack_data(PATH_TO_DATA + '/1', expression, False)
    mat = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    exp = calculate_expression(mat, 'y')
    check_formula_on_synthetic_data(PATH_TO_DATA, exp)
    #
    # x_y_unknown_expressions(mat)
    # calculate_expression(mat, 'y')
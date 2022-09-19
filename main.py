# ===================================================== IMPORTS ====================================================== #


import numpy as np
from typing import Any
import sympy
from sympy import Symbol, simplify, diff, solve, Float, sin, cos  # init_printing, pprint
import mpmath
import pickle
import os
from typing import Tuple


# ===================================================== FORMULA ====================================================== #


def load_expression(expression_path: str):
    with open(expression_path, 'rb') as f:
        expression = pickle.load(f)
    return expression


def calculate_expression(camera_matrix, axis, save_to: str = None):
    # init_printing(use_unicode=True, wrap_line=False, no_global=False)

    # after_image_x = Symbol('after_x')
    # after_image_y = Symbol('after_y')
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

    if axis is 'x':
        rotation_translation_matrix = sympy.Matrix((
            [1, 0, 0, 0],
            [0, cos(alpha), -sin(alpha), 0],
            [0, sin(alpha), cos(alpha), 0],
            [0, 0, 0, 1]
        ))
    elif axis is 'y':
        rotation_translation_matrix = sympy.Matrix((
            [cos(alpha), 0, -sin(alpha), 0],
            [0, 1, 0, 0],
            [sin(alpha), 0, cos(alpha), 0],
            [0, 0, 0, 1]
        ))
    elif axis is 'z':
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

    expression_a = sol[0] / sol[2]
    expression_b = sol[1] / sol[2]

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

    if axis is 'x':
        s11 = solve_a[0].subs(values)
        s12 = solve_a[1].subs(values)
        s11 = mpmath.degrees(simplify(s11))
        s12 = mpmath.degrees(simplify(s12))
        solutions += [s11, s12]
    elif axis is 'y':
        s21 = solve_b[0].subs(values)
        s22 = solve_b[1].subs(values)
        s21 = mpmath.degrees(simplify(s21))
        s22 = mpmath.degrees(simplify(s22))
        solutions += [s22, s21]
    elif axis is 'z':
        s11 = solve_a[0].subs(values)
        s12 = solve_a[1].subs(values)
        s21 = solve_b[0].subs(values)
        s22 = solve_b[1].subs(values)
        s11 = mpmath.degrees(simplify(s11))
        s12 = mpmath.degrees(simplify(s12))
        s21 = mpmath.degrees(simplify(s21))
        s22 = mpmath.degrees(simplify(s22))
        solutions += [s11, s12, s21, s22]

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
        if isinstance(solution, Float) and -2 < solution < 2:
            solution = round(solution, 2)
            solutions.append(solution)
    return solutions


def check_formula_on_synthetic_data(path_to_data: str, ):
    pass


def check_formula_on_optitrack_data():
    pass


# ===================================================== MAIN ========================================================= #


if __name__ == '__main__':
    PATH_TO_DATA = os.getcwd() + '/Dictionary/data/optitrack'
    check_formula_on_optitrack_data(PATH_TO_DATA + '1.csv')

import os
from sympy import Symbol, Matrix, sin, cos, diff, simplify, solve, lambdify, Expr
import numpy as np
import mpmath
import pickle
from typing import Tuple, List, Callable, Dict
import matplotlib.pyplot as plt


def calculate_expression(axis, save_to: str = None) -> List[Expr]:
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
        rotation_translation_matrix = Matrix((
            [1, 0, 0, 0],
            [0, cos(alpha), -sin(alpha), 0],
            [0, sin(alpha), cos(alpha), 0],
            [0, 0, 0, 1]
        ))
    elif axis == 'y':
        rotation_translation_matrix = Matrix((
            [cos(alpha), 0, sin(alpha), 0],
            [0, 1, 0, 0],
            [-sin(alpha), 0, cos(alpha), 0],
            [0, 0, 0, 1]
        ))
    elif axis == 'z':
        rotation_translation_matrix = Matrix((
            [cos(alpha), -sin(alpha), 0, 0],
            [sin(alpha), cos(alpha), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1]
        ))
    else:
        raise ValueError(f'axis must be x, y or z.')
    homogeneous_camera_matrix = Matrix((
        [Fx, 0, Cx, 0],
        [0, Fy, Cy, 0],
        [0, 0, 1, 0]
    ))
    inverse_camera_and_before_point = Matrix((
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
    solutions = []
    for i in solve_a:
        solutions.append(mpmath.degrees(simplify(i)))
    for i in solve_b:
        solutions.append(mpmath.degrees(simplify(i)))
    if save_to is not None:
        try:
            with open(save_to, 'wb') as f:
                pickle.dump(solutions, f)
        except (TypeError, Exception):
            pass
    return solutions


def load_expression(expression_path: str) -> List[Expr]:
    with open(expression_path, 'rb') as f:
        expression = pickle.load(f)
    return expression


def setup(axis: str, path_to_expression: str = None) -> List[Callable]:
    if axis != 'x' and axis != 'y' and axis != 'z':
        raise ValueError('axis must be x, y or z.')
    expression = None
    if path_to_expression is not None:
        try:
            expression = load_expression(path_to_expression)
        except (FileNotFoundError, Exception):
            pass
    if expression is None:
        expression = calculate_expression(axis, path_to_expression)
    x2 = Symbol('after_x')
    y2 = Symbol('after_y')
    x1 = Symbol('before_x')
    y1 = Symbol('before_y')
    Fx = Symbol('Fx')
    Fy = Symbol('Fy')
    Cx = Symbol('Cx')
    Cy = Symbol('Cy')
    inv_Fx = Symbol('inv_Fx')
    inv_Fy = Symbol('inv_Fy')
    inv_Cx = Symbol('inv_Cx')
    inv_Cy = Symbol('inv_Cy')
    solutions = []
    for i in expression:
        solutions.append(lambdify((x1, y1, x2, y2, Fx, Fy, Cx, Cy, inv_Fx, inv_Fy, inv_Cx, inv_Cy), i, 'numpy'))
    return solutions


class Formula:

    functions = {
        'x': setup('x', f'{os.getcwd()}/data/x_rotation.exp'),
        'y': setup('y', f'{os.getcwd()}/data/y_rotation.exp'),
        'z': setup('z', f'{os.getcwd()}/data/z_rotation.exp')
    }

    @staticmethod
    def calculate(vectors: List[Tuple[int, int, int, int]], camera_matrix: np.ndarray,
                  axis: str, decimal_places: int = 1, interval: Tuple[int, int] = (-2, 2)) -> Dict[float, int]:
        if axis != 'x' and axis != 'y' and axis != 'z':
            raise ValueError('axis must be x, y or z.')
        fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
        camera_matrix = np.linalg.inv(camera_matrix)
        ifx, ify, icx, icy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
        solutions = {}
        for func in Formula.functions[axis]:
            for x1, y1, x2, y2 in vectors:
                solution = func(x1, y1, x2, y2, fx, fy, cx, cy, ifx, ify, icx, icy)
                if interval[0] < solution < interval[1]:
                    solution = round(solution, decimal_places)
                    solutions[solution] = solutions.get(solution, 0) + 1
        return solutions

    @staticmethod
    def get_camera_matrix(fov_x=None, fov_y=None, width=None, height=None,
                          fx=None, fy=None, cx=None, cy=None) -> np.ndarray:
        if fx is None:
            if fov_x is None or width is None:
                raise ValueError('since fx wasn\'t given, fov_x and width must be given.')
            fx = width / (2 * np.tan(np.deg2rad(fov_x)))
        if fy is None:
            if fov_y is None or height is None:
                raise ValueError('since fy wasn\'t given, fov_y and height must be given.')
            fy = height / (2 * np.tan(np.deg2rad(fov_y)))
        if cx is None:
            if width is None:
                raise ValueError('since cx wasn\'t given, width must be given.')
            cx = width / 2
        if cy is None:
            if height is None:
                raise ValueError('since cy wasn\'t given, height must be given.')
            cy = height / 2
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    @staticmethod
    def graph_solutions(solutions: Dict[float, int], title: str,
                        save_to: str = None, show: bool = True, bars: int = 10, bar_width: float = 0.35):
        labels = []
        angles_occurrences = []
        for angle, occurrences in sorted(solutions.items(), reverse=True, key=lambda occ: occ[1])[:bars]:
            labels.append(angle)
            angles_occurrences.append(occurrences)
        x = np.arange(len(labels))
        fig, ax = plt.subplots()
        pc_bars = ax.bar(x, angles_occurrences, bar_width, label='PC (ffmpeg h264 encoder)')
        ax.set_ylabel('Number of motion vectors')
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.legend()
        ax.bar_label(pc_bars, padding=3)
        fig.tight_layout()
        if save_to is not None:
            try:
                plt.savefig(save_to)
            except (FileExistsError, Exception) as e:
                print('failed to save figure.')
                print(e)
        if show:
            plt.show()

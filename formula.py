import os
from sympy import Symbol, Matrix, sin, cos, diff, simplify, solve, lambdify, Expr
import numpy as np
import mpmath
import pickle
from typing import Tuple, List, Callable, Dict
import matplotlib.pyplot as plt
from block_matching import BlockMatching
import cv2
from PIL import Image


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
    def calculate(vectors: List[Tuple[int, int, int, int]], camera_matrix: np.ndarray, axis: str,
                  remove_zeros: bool = True, decimal_places: int = 1,
                  interval: Tuple[int, int] = (-2, 2)) -> Dict[float, int]:
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
                    if solution == 0 and remove_zeros:
                        continue
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
                        save_to: str = None, show: bool = True, bars_count: int = 10, bar_width: float = 0.35):
        labels = []
        angles_occurrences = []
        for angle, occurrences in sorted(solutions.items(), reverse=True, key=lambda occ: occ[1])[:bars_count]:
            labels.append(angle)
            angles_occurrences.append(occurrences)
        x = np.arange(len(labels))
        fig, ax = plt.subplots()
        bars = ax.bar(x, angles_occurrences, bar_width)
        ax.set_ylabel('Number of motion vectors')
        ax.set_xlabel('Angle (degrees)')
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.bar_label(bars, padding=3)
        fig.tight_layout()
        if save_to is not None:
            try:
                plt.savefig(save_to)
            except (FileExistsError, Exception) as e:
                print('failed to save figure.')
                print(e)
        if show:
            plt.show()

    @staticmethod
    def run_on_data(path_to_data: str, camera_matrix: np.ndarray, axis: str, angle_step: float, data_name: str = '',
                    save_path: str = None, show: bool = False, interval: Tuple[int, int] = (-50, 50),
                    remove_zeros: bool = True):
        if path_to_data.endswith('.h264') or path_to_data.endswith('.mp4'):
            motion_vectors = BlockMatching.extract_motion_data(path_to_data)
            base_frame = cv2.VideoCapture(path_to_data).read()[1]
        else:
            frames = []
            for i in sorted(os.listdir(path_to_data), key=lambda x: int(x.replace('.png', '').replace('.jpg', ''))):
                frames.append(f'{path_to_data}/{i}')
            motion_vectors = BlockMatching.get_ffmpeg_motion_vectors_with_cache(frames)
            base_frame = cv2.imread(frames[0])
        base_frame = cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = base_frame.shape
        for i, vectors in enumerate(motion_vectors):
            if i % 2 == 1:
                continue
            save_to = f'{save_path}/{i // 2}.png'
            base_image = BlockMatching.draw_motion_vectors(base_frame, vectors, color=(0, 0, 0))
            solutions = Formula.calculate(vectors, camera_matrix, axis, interval=interval, remove_zeros=remove_zeros)
            # print(list(sorted(solutions.items(), key=lambda x: x[1], reverse=True)))
            title = f'{data_name} - {axis.upper()} rotation by {round(angle_step * (1 + i // 2), 2)} degrees'
            Formula.graph_solutions(solutions, title, save_to=save_to, show=show)
            graph = Image.open(save_to, 'r')
            graph_width, graph_height = graph.size
            image_width, image_height = graph_width + frame_width, max(graph_height, frame_height)
            image = Image.new('RGBA', (image_width + 40, image_height), (255, 255, 255, 255))
            image.paste(graph, (20, (image_height - graph_height) // 2))
            image.paste(Image.fromarray(base_image), (40 + graph_width, (image_height - frame_height) // 2))
            image.save(save_to)
            if show:
                cv2.imshow(title, np.asarray(image))
                cv2.waitKey()

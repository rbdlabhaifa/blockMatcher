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


def load_expression(expression_path: str):
    with open(expression_path, 'rb') as f:
        expression = pickle.load(f)
    return expression


def calculate_expression(camera_matrix, axis, save_to: str = None):
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
            [cos(alpha), 0, sin(alpha), 0],
            [0, 1, 0, 0],
            [-sin(alpha), 0, cos(alpha), 0],
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
        if isinstance(solution, Float) and -2 < solution < 2:
            solution = round(solution, 1)
            solutions.append(solution)
    return solutions


def check_formula_on_synthetic_data(path_to_data: str, camera_matrix, axis, debug: bool = False):
    # frames = [f'{path_to_data}/{i}' for i in sorted(os.listdir(path_to_data), key=lambda x: int(x.replace('.png', '')))]
    # motion_vectors = BlockMatching.get_ffmpeg_motion_vectors_with_cache(frames)
    motion_vectors = BlockMatching.extract_motion_data(path_to_data)
    # x2 = Symbol('after_x')
    # x1 = Symbol('before_x')
    # y2 = Symbol('after_y')
    # y1 = Symbol('before_y')
    # funcs = []
    # for i in expression:
    #     funcs.append(lambdify((x1, y1, x2, y2), i, 'numpy'))
    angle = []
    for i, vectors in enumerate(motion_vectors):
        if i % 2 == 1:
            continue
        if debug:
            base_image = cv2.imread(path_to_data + '/0.png')
            base_image = BlockMatching.draw_motion_vectors(base_image, vectors)
            cv2.imshow('', base_image)
            cv2.waitKey()
        ans = Formula.calculate(vectors, camera_matrix, axis, interval=(-20, 20))
        Formula.graph_solutions(ans, f'Experiment - rotation of {5 * (1 + i // 2)} degrees', on_raspi
                                save_to=f'/home/rani/Desktop/graphs/experiment{5 * (1 + i // 2)}.png', show=debug)
    return angle


def check_formula_on_optitrack_data(path_to_data: str, expression, debug: bool = False):
    # frames = [path_to_data + '/' + i for i in sorted(os.listdir(path_to_data), key=lambda x: 66
    # - int(x.replace('.jpg', '')))]
    motion_vectors = BlockMatching.extract_motion_data(path_to_data + '.mp4')
    with open(path_to_data + '.csv', 'r') as file:
        rots = []
        for i in file:
            rots.append(float(i))
    angles = [abs(rots[i] - rots[i - 1]) for i in range(1, len(rots))]
    angles_by_vectors = []
    x2 = Symbol('after_x')
    x1 = Symbol('before_x')
    y2 = Symbol('after_y')
    y1 = Symbol('before_y')
    funcs = []
    for i in expression:
        funcs.append(lambdify((x1, y1, x2, y2), i, 'numpy'))
    for i, vectors in enumerate(motion_vectors):
        # if i % 2 == 1:
        #     continue
        positive_solutions = {}
        negative_solutions = {}
        for x1, y1, x2, y2 in vectors:
            for func in funcs:
                solution = func(x1, y1, x2, y2)
                if solution < -2 or solution > 2:
                    continue
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
        print('negative aver640, 480age solution:', average_neg)
        print('average neg and pos:', average_pos * average_positive - average_neg * average_negative)
        angles_by_vectors.append(negative_solutions[0][0])


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


def to_graph(data: Dict[float, int]):
    """
    Creates histograms from formula results.
    :param data: A dictionary of angles as keys and vector counts as values.
    """
    import matplotlib.pyplot as plt
    x, y = [], []
    for i, j in data.items():
        x += [i]
        y += [j]
    x = list(sorted(x))
    print(x)
    print(y)
    plt.bar(x, y)
    plt.xlabel('Angle')
    plt.ylabel('Vector Count')
    plt.show()

# ===================================================== MAIN ========================================================= #


if __name__ == '__main__':
    p = ''
    fov_x = 60
    fov_y = 60
    width, height = 640, 480
    fx = width / (2 * np.tan(np.deg2rad(fov_x)))
    fy = height / (2 * np.tan(np.deg2rad(fov_y)))
    cx, cy = width / 2, height / 2
    mat = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    # mat = np.array(
    #     [
    #         [629.94662448, 0, 316.23232917],
    #         [0, 629.60772237, 257.64459816],
    #         [0, 0, 1]
    #     ]
    # )
    p = '/home/rani/Downloads'
    # for i in os.listdir(p):
    #     os.rename(p + '/' + i, p + '/' + str(int(i.replace('.png', '')) - 1) + '.png')
    results = check_formula_on_synthetic_data(p, mat, 'x', debug=True)
    #
    # # solutions = {-1.8: 33, -1.9: 31, 0.4: 28, 0.3: 35, -0.0: 2264, -0.2: 3658, -0.1: 500, -0.3: 1224, -0.4: 175, -0.5: 50, -0.6: 9, -1.3: 2, -1.2: 1, -0.7: 5, -0.9: 1, -0.8: 1, -1.0: 1, 0.5: 35, 0.8: 29, -1.4: 1, -1.1: 1, 0.9: 1, 1.5: 1, -2.0: 1}
    # for angle in results:
    #     Formula.graph_solutions(angle, title='VR Video - x rotation by 0.2 degrees', save_to='graph.png')


    # j =0
    # for i in (sorted(os.listdir('/home/rani/Pictures'), key=lambda x: int(x[-6:-4]))):
    #     im = cv2.imread('/home/rani/Pictures/' + i)[120:480, 80:720]
    #     cv2.imwrite('/home/rani/Pictures/' + str(j) + '.png', im)
    #     j += 1
    # expression = load_expression('600x600-360x.exp')
    # check_formula_on_synthetic_data(, exp, True)
    # check_formula_on_synthetic_data('/home/rani/old', expression, True)
    # path_to_data = ('/home/rani/Documents/cut')
    # path_to_data2 = ('/home/rani/old')
    #
    # suffix = '.png'
    # frames = [f'{path_to_data}/{i}' for i in
    #           sorted(os.listdir(path_to_data), key=lambda x: int(x.replace(suffix, '')))]
    # motion_vectors = BlockMatching.get_ffmpeg_motion_vectors_with_cache(frames)
    # for i, vectors in enumerate(motion_vectors):
    #     if suffix == '.png':
    #         if i % 2 == 1:
    #             continue
    #         print('angle=', 0.1 * ((i // 2) + 1))
    #     base_image = cv2.imread(path_to_data + '/0' + suffix)
    #     base_image = BlockMatching.draw_motion_vectors(base_image, vectors)
    #     base_image1 = cv2.imread(path_to_data2 + '/0' + suffix)
    #     base_image1 = BlockMatching.draw_motion_vectors(base_image1, vectors)
    #     cv2.imshow('', base_image)
    #     cv2.imshow('11', base_image1)
    #     cv2.waitKey()
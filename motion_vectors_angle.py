from sympy import *
import numpy as np
import mpmath
from typing import Tuple


def calculate_angle(camera_matrix:  np.ndarray, vector: Tuple[int, int, int, int]):

    cam_mat = camera_matrix

    before_image_p = (vector[0], vector[1])
    after_image_p = (vector[0], vector[1])

    init_printing(use_unicode=True, wrap_line=False, no_global=False)
    inv_cam_mat = np.linalg.inv(cam_mat)
    to_print = False

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

    expression_a = (((Cx*sin(alpha) * (inv_Fy*before_image_y+inv_Cy) + Cx*cos(alpha) + Fx*(inv_Fx*before_image_x+inv_Cx))/
                     (cos(alpha)+sin(alpha)*(inv_Fy*before_image_y+inv_Cy))) - after_image_x)**2
    expression_b = ((((-Fy)*sin(alpha)+(inv_Fy*before_image_y+inv_Cy)*(Fy*cos(alpha)+Cy*sin(alpha))+Cy*cos(alpha))/
                     (cos(alpha)+sin(alpha)*(inv_Fy*before_image_y+inv_Cy))) - after_image_y)**2

    values = [(Cx,cam_mat[0][2]),(Cy,cam_mat[1][2]),(Fx,cam_mat[0][0]),(Fy,cam_mat[1][1]),(inv_Fx,inv_cam_mat[0][0]),(inv_Fy,inv_cam_mat[1][1]),
              (inv_Cx,inv_cam_mat[0][2]),(inv_Cy,inv_cam_mat[1][2]),(after_image_x,after_image_p[0]),
              (after_image_y,after_image_p[1]),(before_image_x,before_image_p[0]),(before_image_y,before_image_p[1])]

    s1 = expression_a.subs(values)
    s2 = expression_b.subs(values)

    dif_a = diff(expression_a,alpha)
    dif_b = diff(expression_b,alpha)

    simple_a = simplify(dif_a)
    simple_b = simplify(dif_b)

    solve_a = solve(simple_a,alpha)
    solve_b = solve(simple_b,alpha)

    s11= solve_a[0].subs(values)
    s12= solve_a[1].subs(values)
    s13= solve_a[2].subs(values)
    s14= solve_a[3].subs(values)
    s21 = solve_b[0].subs(values)
    s22 = solve_b[1].subs(values)

    s11 = simplify(s11)
    s12 = simplify(s12)
    s13 = simplify(s13)
    s14 = simplify(s14)
    s21 = simplify(s21)
    s22 = simplify(s22)

    s11 = mpmath.degrees(s11)
    s12 =  mpmath.degrees(s12)
    s13 =  mpmath.degrees(s13)
    s14 =  mpmath.degrees(s14)
    s21 =  mpmath.degrees(s21)
    s22 =  mpmath.degrees(s22)

    if to_print is True:
        print('The first solution is : ')
        pprint(s11)
        print('The second solution is : ')
        pprint(s12)
        print('The third solution is : ')
        pprint(s13)
        print('The fourth solution is : ')
        pprint(s14)
        print('The fifth solution is : ')
        pprint(s21)
        print('The sixth solution is : ')
        pprint(s22)
    ret = np.array([s11,s12,s13,s14,s21,s22])
    index = []
    for idx,i in enumerate(ret):
        if type(i) is not Float or i < 0.:
            index.append(idx)
    new_ret = np.delete(ret,index)
    print(new_ret)
    return new_ret


def calculate_camera_matrix(width: int, height: int, fov_x: int, fov_y: int):
    return np.array([
        [width / (2 * np.tan(np.deg2rad(fov_x))), 0, width / 2],
        [0, height / (2 * np.tan(np.deg2rad(fov_y))), height / 2],
        [0, 0, 1]
    ])


if __name__ == '__main__':
    size = (1000, 1000)
    fov_x = 50
    fov_y = 50
    cam_mat = np.array([[size[0] / (2 * np.tan(np.deg2rad(fov_x))), 0, size[0] / 2],
                        [0, size[1] / (2 * np.tan(np.deg2rad(fov_y))), size[1] / 2],
                        [0, 0, 1]])
    before_image_p = [9,1]
    after_image_p = [8,1]
    calculate_angle(cam_mat, before_image_p, after_image_p)

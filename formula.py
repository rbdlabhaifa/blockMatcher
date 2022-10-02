# def calculate_expression():
#     after_image_x = Symbol('after_x')
#     after_image_y = Symbol('after_y')
#     before_image_x = Symbol('before_x')
#     before_image_y = Symbol('before_y')
#
#     Fx = Symbol('Fx')
#     Fy = Symbol('Fy')
#     Cx = Symbol('Cx')
#     Cy = Symbol('Cy')
#
#     inv_Fx = Symbol('inv_Fx')
#     inv_Fy = Symbol('inv_Fy')
#     inv_Cx = Symbol('inv_Cx')
#     inv_Cy = Symbol('inv_Cy')
#     alpha = Symbol('alpha')
#
#     if axis == 'x':
#         rotation_translation_matrix = sympy.Matrix((
#             [1, 0, 0, 0],
#             [0, cos(alpha), -sin(alpha), 0],
#             [0, sin(alpha), cos(alpha), 0],
#             [0, 0, 0, 1]
#         ))
#     elif axis == 'y':
#         rotation_translation_matrix = sympy.Matrix((
#             [cos(alpha), 0, sin(alpha), 0],
#             [0, 1, 0, 0],
#             [-sin(alpha), 0, cos(alpha), 0],
#             [0, 0, 0, 1]
#         ))
#     elif axis == 'z':
#         rotation_translation_matrix = sympy.Matrix((
#             [cos(alpha), -sin(alpha), 0, 0],
#             [sin(alpha), cos(alpha), 0, 0],
#             [0, 0, 1, 0],
#             [0, 0, 1, 1]
#         ))
#     else:
#         raise ValueError(f'axis must be x, y or z.')
#
#     homogeneous_camera_matrix = sympy.Matrix((
#         [Fx, 0, Cx, 0],
#         [0, Fy, Cy, 0],
#         [0, 0, 1, 0]
#     ))
#     inverse_camera_and_before_point = sympy.Matrix((
#         [inv_Fx * before_image_x + inv_Cx],
#         [inv_Fy * before_image_y + inv_Cy],
#         [1],
#         [1]
#     ))
#     sol = homogeneous_camera_matrix * rotation_translation_matrix * inverse_camera_and_before_point
#
#     expression_a = ((sol[0] / sol[2]) - after_image_x) ** 2
#     expression_b = ((sol[1] / sol[2]) - after_image_y) ** 2
#
#     dif_a = diff(expression_a, alpha)
#     dif_b = diff(expression_b, alpha)
#
#     simple_a = simplify(dif_a)
#     simple_b = simplify(dif_b)
#     solve_a = solve(simple_a, alpha)
#     solve_b = solve(simple_b, alpha)
#
#     inverse_camera_matrix = np.linalg.inv(camera_matrix)
#
#     values = {
#         Fx: camera_matrix[0, 0],
#         Fy: camera_matrix[1, 1],
#         Cx: camera_matrix[0, 2],
#         Cy: camera_matrix[1, 2],
#         inv_Fx: inverse_camera_matrix[0, 0],
#         inv_Fy: inverse_camera_matrix[1, 1],
#         inv_Cx: inverse_camera_matrix[0, 2],
#         inv_Cy: inverse_camera_matrix[1, 2],
#     }
#
#     solutions = []
#
#     for i in solve_a:
#         solutions.append(mpmath.degrees(simplify(i.subs(values))))
#     for i in solve_b:
#         solutions.append(mpmath.degrees(simplify(i.subs(values))))
#
#     if save_to is not None:
#         try:
#             with open(save_to, 'wb') as f:
#                 pickle.dump(solutions, f)
#         except (TypeError, Exception):
#             pass
#
#     return solutions


class Formula:

    def __init__(self):
        pass

    def load_expressions(self):
        pass

    def save_expressions(self):
        pass

    def check(self):
        pass

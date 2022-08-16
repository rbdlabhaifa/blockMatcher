import numpy as np
from python_block_matching import *
from mv_dictionary import MVMapping
import cv2
from syntetic import generate_pictures_2_angles


def calculate_rotation(mv):
    fov_x = 60
    res_x = 480
    vector_length = abs(mv[2] - mv[0])
    conv_value = (np.tan(np.deg2rad(fov_x / 2)) / (res_x / 2))
    vector_length *= conv_value
    mv = list(mv)
    mv[0] -= res_x/2
    mv[2] -= res_x/2
    a = np.sqrt((mv[0] * conv_value) ** 2 + 1)
    b = np.sqrt((mv[2] * conv_value) ** 2 + 1)
    c = vector_length
    return np.rad2deg(np.arccos((-(c ** 2) + a ** 2 + b ** 2) / (2 * a * b)))


def try_ego_rotation_dicts():
    im_shape = [360, 360]
    square_size = 32
    mv_dict = MVMapping('trained dicts/chess_plane_ego_rot_0-8_steps.pickle')
    while True:
        angle, step = eval(input('enter angle and step').replace(' ', ','))
        chess = np.zeros((im_shape[0] * 4, int(im_shape[1] * 4), 3), dtype=np.uint8)
        for i in range(int(chess.shape[0] / square_size)):
            for j in range(int(chess.shape[1] / square_size)):
                chess[square_size * j:square_size * (j + 1), square_size * i:square_size * (i + 1)] = np.random.randint(
                    0,
                    256,
                    3,
                    np.uint8)
        f1, f2 = generate_pictures_2_angles(chess, angle, angle + step, im_shape)
        print('dictionary found: ', mv_dict[BlockMatching.get_motion_vectors(f2, f1)])


if __name__ == '__main__':
    m = MVMapping()
    m.load_from('/home/rani/PycharmProjects/blockMatcher/Dictionary Testing/trained dicts/detailed counter clockwise')
    m.load_from('/home/rani/PycharmProjects/blockMatcher/Dictionary Testing/trained dicts/detailed clockwise')
    m.load_from('/home/rani/PycharmProjects/blockMatcher/Dictionary Testing/trained dicts/neutral clockwise')
    m.load_from('/home/rani/PycharmProjects/blockMatcher/Dictionary Testing/trained dicts/neutral counter clockwise')
    m.save_to('/home/rani/PycharmProjects/blockMatcher/Dictionary Testing/trained dicts/markers - full')

    # image = cv2.imread('projection/image0.0.png')
    # for i in range(0, 100, 5):
    #     im = cv2.imread(f'projection/image{i / 10}.png')
    #     mvs = BlockMatching.get_motion_vectors(image, im)
    #     mvs = list(filter(lambda p: p[3] - p[1] == 0, mvs))
    #     f = BMFrame(im)
    #     f.draw_motion_vector(mvs, (0, 0, 255), 1)
    #     rot = np.max([calculate_rotation(mv) for mv in mvs])
    #     print(rot, i / 10)
    #
    #     f.show()

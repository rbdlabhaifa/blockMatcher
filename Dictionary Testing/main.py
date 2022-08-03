import numpy as np
# from syntetic import generate_pictures_2_angles
from python_block_matching import *
from mv_dictionary import MVMapping


def calculate_rotation(mv):
    fov_x = 60
    res_x = 480
    vector_length = abs(mv[2] - mv[0])
    conv_value = (np.tan(fov_x / 2) / (res_x / 2))
    vector_length *= conv_value
    mv = list(mv)
    mv[0] -= res_x/2
    mv[2] -= res_x/2
    a = np.sqrt((mv[0] * conv_value) ** 2 + 1)
    b = np.sqrt((mv[2] * conv_value) ** 2 + 1)
    c = vector_length
    print("Lengths: " ,a,b,c)
    print("coords", mv[0]*(conv_value), mv[2]*(conv_value))
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
        # a = BMFrame(f1)
        # a.draw_motion_vector(BlockMatching.get_motion_vectors(f2, f1), (0, 255, 0), 2)
        # a.show()
        # BMFrame(f2).show()
        print('dictionary found: ', mv_dict[BlockMatching.get_motion_vectors(f2, f1)])


import cv2
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':

    # mvd = MVMapping()
    image = cv2.imread('projection/image0.0.png')
    for i in range(0, 100, 5):
        im = cv2.imread(f'projection/image{i / 10}.png')
        mvs = BlockMatching.get_motion_vectors(image, im,
                                               current_frame_mb=[(im.shape[0] // 2 - 8, im.shape[1] // 2 - 8, 16, 16)])
        f = BMFrame(im)
        f.draw_motion_vector(mvs, (0, 0, 255), 1)
        print(mvs, 0, 0, i / 10, calculate_rotation(mvs[0]))
        f.show()
    #     mvd[mvs] = (0, 0, i / 10)
    # mvd.save_to('trained dicts/gradient')

    # vid = BMVideo('created.mp4')
    # mvd = MVMapping('trained dicts/gradient')
    # for i in range(vid.get_frame_count() - 1):
    #     frame1, frame2 = vid[i], vid[i + 1]
    #     mvs = BlockMatching.get_motion_vectors(frame1.base_image, frame2.base_image)
    #     frame1.draw_motion_vector(mvs, (200, 200, 100), 1)
    #     cv2.imwrite(f'frame{i}.png', frame1.base_image)

    # try_ego_rotation_dicts()
    # mv_dict = MVMapping('trained dicts/square')
    #
    # vid = BMVideo([f'rot/Frame{i}.jpeg' for i in range(90)])
    # for i in range(vid.get_frame_count() - 1):
    #     f1, f2 = vid[i], vid[i + 1]
    #     mvs = BlockMatching.get_motion_vectors(f2.base_image, f1.base_image)
    #     translation = mv_dict[mvs]
    #     mvs = MVMapping.remove_zeroes(mvs)
    #     f1.draw_motion_vector(mvs, (0, 255, 0), 1)
    #     f1.show()
    #     mv = mvs[0]
    #     for m in range(len(mvs)):
    #         if abs(mv[2] - mv[0]) < abs(mvs[m][2] - mvs[m][0]):
    #             mv = mvs[m]
    #     print(mv)
    #     angle = MVMapping.calculate_camera_x_rotation(mv, f1.base_image.shape[:2][::-1], 45)
    #     print(angle)

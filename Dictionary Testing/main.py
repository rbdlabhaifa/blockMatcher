import numpy as np
from syntetic import generate_pictures_2_angles
from python_block_matching import *
from mv_dictionary import MVMapping


def test_ego_rotation_dicts():
    im_shape = [480, 480]
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
    mv_dict = MVMapping('trained dicts/square')

    vid = BMVideo([f'rot/Frame{i}.jpeg' for i in range(90)])
    for i in range(vid.get_frame_count() - 1):
        f1, f2 = vid[i], vid[i + 1]
        mvs = BlockMatching.get_motion_vectors(f2.base_image, f1.base_image)
        translation = mv_dict[mvs]
        mvs = MVMapping.remove_zeroes(mvs)
        f1.draw_motion_vector(mvs, (0, 255, 0), 1)
        f1.show()
        mv = mvs[0]
        for m in range(len(mvs)):
            if abs(mv[2] - mv[0]) < abs(mvs[m][2] - mvs[m][0]):
                mv = mvs[m]
        print(mv)
        angle = MVMapping.calculate_camera_x_rotation(mv, f1.base_image.shape[:2][::-1], 45)
        print(angle)

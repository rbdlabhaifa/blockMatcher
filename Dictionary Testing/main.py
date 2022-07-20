import numpy as np

from python_block_matching import *
from mv_dictionary import MVMapping


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
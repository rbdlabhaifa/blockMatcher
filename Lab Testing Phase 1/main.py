from mv_dictionary import MVMapping
from python_block_matching import BlockMatching, BMFrame, BMVideo
from data_generator import DataGenerator
import cv2
import numpy as np

if __name__ == '__main__':
    url = "Benchmark_Pictures/Image0.png"
    img = cv2.imread(url)
    mv_dict = MVMapping("dict.pickle")
    # for flip in range(-1, 2, 2):
    #     for i in range(10, 110, 10):
    #         translation = (flip * i, 0, 0)
    #         f1, f2 = DataGenerator.generate_translation([360, 360], url, translation[:2])
    #         mv_dict[[*BlockMatching.get_motion_vectors(f2, f1)]] = translation
    #         translation = (0, flip * i, 0)
    #         f1, f2 = DataGenerator.generate_translation([360, 360], url, translation[:2])
    #         mv_dict[[*BlockMatching.get_motion_vectors(f2, f1)]] = translation
    #         translation = (flip * i, flip * i, 0)
    #         f1, f2 = DataGenerator.generate_translation([360, 360], url, translation[:2])
    #         mv_dict[[*BlockMatching.get_motion_vectors(f2, f1)]] = translation
    #         translation = (-flip * i, flip * i, 0)
    #         f1, f2 = DataGenerator.generate_translation([360, 360], url, translation[:2])
    #         mv_dict[[*BlockMatching.get_motion_vectors(f2, f1)]] = translation
    #
    # for i in range(0, 360):
    #     f1, f2 = DataGenerator.generate_rotation([360, 360], url, i)
    #     mv_dict[[*BlockMatching.get_motion_vectors(f2, f1)]] = [0, 0, i]
    #
    # mv_dict.save_to("dict.pickle")
    is_trans = input("T = Trans, R = Rotation") == "T"

    if is_trans:
        x_trans = input("Enter translation on the X-axis. To Exit enter none number value")
        y_trans = input("Enter translation on the Y-axis. To Exit enter none number value")
    else:
        angle = input("Enter Angle. To Exit enter none number value")

    while (is_trans and (x_trans.isnumeric() and y_trans.isnumeric())) or angle.isnumeric():
        if is_trans:
            translation = [int(x_trans), int(y_trans)]
            f1, f2 = DataGenerator.generate_translation([360, 360], url, translation)
            prediction = mv_dict[[*BlockMatching.get_motion_vectors(f2, f1)]]
            print(f"User Entered: [{x_trans},{y_trans}], Dictionary returned: {prediction}")

        else:
            f1, f2 = DataGenerator.generate_rotation([360,360], url, int(angle))
            prediction = mv_dict[[*BlockMatching.get_motion_vectors(f2, f1)]]
            print(f"User Entered: {angle}, Dictionary returned: {prediction}")

        is_trans = input("T = Trans, R = Rotation") == "T"

        if is_trans:
            x_trans = input("Enter translation on the X-axis. To Exit enter none number value")
            y_trans = input("Enter translation on the Y-axis. To Exit enter none number value")
        else:
            angle = input("Enter Angle. To Exit enter none number value")
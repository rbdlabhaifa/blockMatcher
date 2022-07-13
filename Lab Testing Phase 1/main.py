from mv_dictionary import MVDict
from python_block_matching import BlockMatching, BMFrame, BMVideo
from data_generator import DataGenerator
import cv2
import numpy as np

if __name__ == '__main__':
    url = "Benchmark_Pictures/Black.png"
    img = cv2.imread(url)
    mv_dict = MVDict()
    for flip in range(-1, 2, 2):
        for i in range(10, 110, 10):
            translation = [flip * i, 0]
            f1, f2 = DataGenerator.generate_movement([360, 360], url, translation)
            mv_dict[BlockMatching.get_motion_vectors(f2, f1)] = translation
            translation = [0, flip * i]
            f1, f2 = DataGenerator.generate_movement([360, 360], url, translation)
            mv_dict[BlockMatching.get_motion_vectors(f2, f1)] = translation
            translation = [flip * i, flip * i]
            f1, f2 = DataGenerator.generate_movement([360, 360], url, translation)
            mv_dict[BlockMatching.get_motion_vectors(f2, f1)] = translation
            translation = [-flip * i, flip * i]
            f1, f2 = DataGenerator.generate_movement([360, 360], url, translation)
            mv_dict[BlockMatching.get_motion_vectors(f2, f1)] = translation

    x_trans = input("Enter translation on the X-axis. To Exit enter none number value")
    y_trans = input("Enter translation on the Y-axis. To Exit enter none number value")

    while x_trans.isnumeric() and y_trans.isnumeric():
        translation = [x_trans, y_trans]
        f1, f2 = DataGenerator.generate_movement([360, 360], url, translation)
        prediction = mv_dict[BlockMatching.get_motion_vectors(f2, f1)]
        print(f"User Entered: [{x_trans},{y_trans}], Dictionary returned: {prediction}")

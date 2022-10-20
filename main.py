import cv2
import numpy as np
import os
from block_matching import BlockMatching
from formula import *


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
        motion_vectors = BlockMatching.get_ffmpeg_motion_vectors(frames)
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


if __name__ == '__main__':
    from mvextractor.videocap import VideoCap

    # 640 x 480
    # camera_matrix = np.array([[647.81841563, 0, 335.8814632],
    #                           [0, 645.9438274, 225.99776891001],
    #                           [0, 0, 1]])
    # 1280 x 720
    # camera_matrix = np.array([[1374.469951742393, 0, 638.379203340978],
    #                           [0, 1380.753021831714, 374.04638859868663],
    #                           [0, 0, 1]])
    # 960 x 720
    camera_matrix = np.array([
        [907.15425425, 0, 478.97775485],
        [0, 907.55348533, 346.90456115],
        [0, 0, 1]
    ])
    vectors_cap = VideoCap()
    vectors_cap.open('/home/rani/PycharmProjects/blockMatcher/data/drone/1.mp4')
    was_read, frame, vectors, frame_type, _ = vectors_cap.read()
    i = 0
    total = 0
    print(frame.shape)
    all_sols = {}
    while was_read:
        # if frame_type == 'I':
        #     print('i frame')
        #     was_read, frame, vectors, frame_type, _ = vectors_cap.read()
        #     continue
        mvs = vectors[:, 3:7]
        sols = Formula.calculate(mvs, camera_matrix, 'y', decimal_places=2, interval=(-2, 2), remove_zeros=True)
        # Formula.graph_solutions(sols, '', bars_count=5, show=True)
        was_read, frame, vectors, frame_type, _ = vectors_cap.read()
        # frame = BlockMatching.draw_motion_vectors(frame, mvs)
        # cv2.imshow('', frame)
        # cv2.waitKey()
        if len(sols):
            max_sol = abs(max(sols.items(), key=lambda x: x[1])[0])
            total += max_sol
            all_sols[i] = total
        i += 1
        print(i, total)
    print(all_sols)
    import matplotlib.pyplot as plt

    Year = [i for i in all_sols.keys()]
    Unemployment_Rate = [i for i in all_sols.values()]

    plt.plot(Year, Unemployment_Rate)
    # plt.title('Unemployment Rate Vs Year')
    # plt.xlabel('sum of angles')
    # plt.ylabel('Unemployment Rate')
    plt.show()

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
    vectors_cap = VideoCap()
    vectors_cap.open('/home/rani/PycharmProjects/data/webcam/2.h264')

    was_read, frame, vectors, frame_type, _ = vectors_cap.read()
    i = 0
    while was_read:
        if frame_type == 'I':
            print(i)
            # cv2.imwrite(f'images/{i}.png', frame)
            was_read, frame, vectors, frame_type, _ = vectors_cap.read()
            i += 1
            continue
        # mvs = vectors[:, 3:7]
        # print(mvs)
        # if len(mvs):
            # frame = BlockMatching.draw_motion_vectors(frame, mvs)
        # cv2.imwrite(f'images/{i}.png', frame)
        was_read, frame, vectors, frame_type, _ = vectors_cap.read()

        i += 1
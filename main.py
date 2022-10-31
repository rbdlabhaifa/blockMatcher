import os
import threading
import time
from formula import Formula
from mvextractor.videocap import VideoCap
import cv2
from block_matching import BlockMatching
import numpy as np
from djitellopy import Tello


def main_drone():
    cont = True
    def film():
        f = tello.get_frame_read()
        i = 0
        w = 0.5 / 30
        while cont:
            frame = f.frame
            cv2.imwrite(f'data/drone/8/{i}.png', frame)
            i += 1
            cv2.imshow('', frame)
            cv2.waitKey(1)
            time.sleep(w)

    thread = threading.Thread(target=film)
    tello = Tello()

    tello.connect()
    tello.streamon()
    thread.start()
    tello.takeoff()
    time.sleep(1)
    tello.move_up(100)
    # for i in range(8):
    time.sleep(0.5)
    for i in range(11):
        tello.send_rc_control(0, 0, 0, 15)
        time.sleep(6)
    cont = False
    thread.join()
    tello.streamoff()
    tello.land()


def rename(path):
    p = path
    m = 2
    ar = os.listdir(p)
    ar.remove('info.txt')
    for i in sorted(ar, key=lambda x: int(x[:1])):
        os.rename(p + '/' + i, p + '/' + str(int(i[:1]) - m) + (i[1:] if len(i) > 1 else ''))


def form():
    camera_matrix = np.array([
        [907.15425425, 0, 478.97775485],
        [0, 907.55348533, 346.90456115],
        [0, 0, 1]
    ])
    p = '/home/rani/PycharmProjects/blockMatcher/data/drone/8_4.mp4'
    # BlockMatching.get_ffmpeg_motion_vectors(
    #     [f'{p}/{i}.png' for i in range(3, 1253, 4)], save_to='/home/rani/PycharmProjects/blockMatcher/data/drone/8_4.mp4',
    #     repeat_first_frame=False
    # )

    vectors_cap = VideoCap()
    vectors_cap.open(p)
    was_read, frame, vectors, frame_type, _ = vectors_cap.read()
    i = 0
    total = 0
    print(frame.shape)
    all_sols = {}
    c = 0
    while was_read:
        if frame_type == 'I':
            print('i frame')
            c += 1
            was_read, frame, vectors, frame_type, _ = vectors_cap.read()
            continue
        mvs = vectors[:, 3:7]

        sols = Formula.calculate(mvs, camera_matrix, 'y', decimal_places=5, interval=(-10, 10), remove_zeros=True)
        # Formula.gr aph_solutions(sols, '', bars_count=5, show=True)x
        was_read, frame, vectors, frame_type, _ = vectors_cap.read()
        # frame = BlockMatching.draw_motion_vectors(frame, mvs)
        # cv2.imshow('', frame)
        # cv2.waitKey()
        if len(sols):
            max_sol = max(sols.items(), key=lambda x: x[1])[0]
            total += max_sol
            all_sols[i] = total
        i += 1
        print(i, total)
    print(all_sols)
    print(c)
    import matplotlib.pyplot as plt

    Year = [i for i in all_sols.keys()]
    Unemployment_Rate = [i for i in all_sols.values()]

    plt.plot(Year, Unemployment_Rate)
    # plt.title('Unemployment Rate Vs Year')
    # plt.xlabel('sum of angles')
    # plt.ylabel('Unemployment Rate')
    plt.show()




if __name__ == '__main__':
    # main_drone()
    rename('/home/rani/PycharmProjects/blockMatcher/data/optitrack')
    # form()

import time, cv2
from threading import Thread
from djitellopy import Tello

tello = Tello()

tello.connect()

keepRecording = True
tello.streamon()
frame_read = tello.get_frame_read()

def videoRecorder():
    framenum = 0
    while keepRecording:
        cv2.imwrite(f'/home/rani/PycharmProjects/blockMatcher/data/drone/3/{framenum}.png', frame_read.frame)
        framenum += 1
        time.sleep(1 / 30)

recorder = Thread(target=videoRecorder)

# tello.streamon()
tello.takeoff()
# time.sleep(3)
# recorder.start()

tello.rotate_clockwise(20)
tello.land()

keepRecording = False
recorder.join()

# 360 error of 1 deg
# 5 error of 0.45 deg
# 10 error of 1.3 deg
# 15 error of 1.3 deg
# 20 error of 1.4 deg

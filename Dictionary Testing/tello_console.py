import time
from djitellopy import Tello
import cv2
from mv_dictionary import MVMapping
from python_block_matching import BlockMatching

# Constants
REFERENCE_ANGLE = 0  # angle of the reference frame
HEIGHT = 25
HEIGHT_TOLERANCE = 5
ANGLE_SET = (5, 10, 15, 20, 25, 30)


# Util Functions
def rotate_and_wait(drone: Tello, angle, verbose=False):
    if angle > 0:
        out = drone.send_command_with_return(f"cw {angle}")
    else:
        out = drone.send_command_with_return(f"ccw {angle}")
    if verbose:
        return out


def set_height(drone: Tello, height, tolerance, verbose=False):
    while abs(tello.get_height() - height) > tolerance:
        error = tello.get_height() - height
        if error < 0:
            print(tello.send_command_with_return(f"up {abs(error)}"))
        else:
            print(tello.send_command_with_return(f"down {error}"))

        print(tello.get_height(), error)
        time.sleep(1)


# Startup Routine
tello = Tello()
tello.connect()
tello.streamon()
frame_capture = tello.get_frame_read()

# Initialize Variables
mv_dict = MVMapping()
flips = []

# Adjust Height
tello.send_command_with_return("takeoff")
set_height(tello, HEIGHT, HEIGHT_TOLERANCE)

ref_frame = frame_capture.frame

# Take Pictures
for flip in (1, -1):
    frames = []
    for angle in ANGLE_SET:
        rotate_and_wait(tello,5*flip)
        print(f"Aim camera at angle {flip * angle}")
        frames.append(frame_capture.frame)
    if flip == -1:
        tello.rotate_clockwise(30)
    else:
        tello.rotate_counter_clockwise(30)
    time.sleep(1)

    flips.append(frames)

print("Done Taking Pictures")
tello.land()
exit()
# Compute Dictionary
print("Adding entries to dictionary")
for flip in range(2):
    for frame_index in range(len(frames)):
        mv_dict[BlockMatching.get_motion_vectors(flips[flip][frame_index], ref_frame)] = (-1,1)[flip] * (5, 10, 15, 20, 25, 30)[
            frame_index]

save = input("Dictionary Complete! Save n/y?")

if save == 'q':
    name = input("Enter Name Of File To Save In")
    mv_dict.save_to("trained dicts/" + name + ".pickle")

# Testing Dictionary
while True:
    num = input("Enter Degree")
    while num:
        if num > 0:
            tello.rotate_clockwise(num)
        else:
            tello.rotate_counter_clockwise(num)
        frame = frame_capture.frame
        print(f"Dict Returned: {mv_dict[BlockMatching.get_motion_vectors(frame, ref_frame)]}")
        if num < 0:
            tello.rotate_clockwise(num)
        else:
            tello.rotate_counter_clockwise(num)

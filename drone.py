"""

This script rotates a Tello drone that is registered as a rigid body in Motive.
The drones video stream is saved to images and OptiTrack rotation and translation is saved to a csv file.

"""


import ctypes
from djitellopy import Tello
import cv2
import os
import sys


# Parameters.
PROJECT_PATH = sys.argv[1].encode()             # Path to the .ttp project.
OUTPUT_DIRECTORY_PATH = sys.argv[2].encode()    # Path to a directory to save the output to.

# FIXME:
LIBRARY_PATH = 'C:/Users/fares/Desktop/ben jobi/python/NPTrackingToolsx64.dll' if len(sys.argv) < 4 else sys.argv[3]

# Load the Motive API library and setup relevant functions.
motive = ctypes.cdll.LoadLibrary(LIBRARY_PATH)
motive.TT_LoadProject.argtype = (ctypes.c_char_p,)
motive.TT_LoadProject.argtype = (ctypes.c_char_p,)
motive.TT_RigidBodyName.argtype = (ctypes.c_int,)
motive.TT_RigidBodyName.restype = ctypes.c_char_p
motive.TT_RigidBodyLocation.argtype = (ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))
motive.TT_CameraName.argtype = (ctypes.c_int,)
motive.TT_CameraName.restype = ctypes.c_char_p
motive.TT_CameraFrameRate.argtype = (ctypes.c_int,)
motive.TT_CameraExposure.argtype = (ctypes.c_int,)
motive.TT_CameraIntensity.argtype = (ctypes.c_int,)
motive.TT_CameraVideoType.argtype = (ctypes.c_int,)
motive.TT_IsRigidBodyTracked.argtype = (ctypes.c_int,)
motive.TT_RigidBodyResetOrientation.argtype = (ctypes.c_int,)


def check_return_code(return_code: int):
    """
    Handle the Motive APIs return codes.

    :param return_code: The NPRESULT code that was returned in an operation.
    """
    if return_code == 0:
        return print('Success.')
    elif return_code == 1:
        print('Error: File not found.')
    elif return_code == 2:
        print('Error: Load failed.')
    elif return_code == 3:
        print('Error: Operation failed.')
    elif return_code == 8:
        print('Error: Invalid file.')
    elif return_code == 9:
        print('Error: Invalid calibration file.')
    elif return_code == 10:
        print('Error: Unable to initialize.')
    elif return_code == 11:
        print('Error: Invalid license.')
    elif return_code == 14:
        print('Error: No frames available.')
    else:
        print(f'Unknown error code, ({return_code}).')
    while motive.TT_Shutdown() != 0:
        print('Trying to shutdown...')
    print('Done.')


# Initialize the OptiTrack system.
check_return_code(motive.TT_Initialize())

# Load the .ttp project.
motive.TT_LoadProject(ctypes.c_char_p(PROJECT_PATH))

# List detected cameras.
print('Cameras:')
for i in range(motive.TT_CameraCount()):
    print(f'    Camera #{i}:')
    i = ctypes.c_int(i)
    print(f'        Name:', motive.TT_CameraName(i).decode())
    print(f'        FPS:', motive.TT_CameraFrameRate(i))
    print(f'        Exposure:', motive.TT_CameraExposure(i))
    print(f'        Intensity:', motive.TT_CameraIntensity(i))
    camera_mode = motive.TT_CameraVideoType(i)
    if camera_mode == 0:
        print(f'        Intensity: Segment Mode')
    elif camera_mode == 1:
        print(f'        Intensity: Grayscale Mode')
    elif camera_mode == 2:
        print(f'        Intensity: Object Mode')
    elif camera_mode == 4:
        print(f'        Intensity: Precision Mode')
    elif camera_mode == 6:
        print(f'        Intensity: MJPEG Mode')
    else:
        print(f'        Intensity: Unknown Mode')

# List all defined rigid bodies.
print('Rigid Bodies:')
for i in range(motive.TT_RigidBodyCount()):
    print(f'    Rigid Body #{i}:')
    i = ctypes.c_int(i)
    print(f'        Name:', motive.TT_RigidBodyName(i).decode())
    x, y, z = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
    qx, qy, qz, qw = ctypes.c_float(), ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
    pitch, yaw, roll = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
    motive.TT_RigidBodyLocation(i, ctypes.byref(x), ctypes.byref(y), ctypes.byref(z), ctypes.byref(qx),
                                ctypes.byref(qy), ctypes.byref(qz), ctypes.byref(qw), ctypes.byref(pitch),
                                ctypes.byref(yaw), ctypes.byref(roll))
    print(f'        Position:', (x.value, y.value, z.value))
    print(f'        Orientation:', (pitch.value, yaw.value, roll.value), (qx.value, qy.value, qz.value, qw.value))

# Files to keep the OptiTrack's data.
# rotation_file = open(f'{OUTPUT_DIRECTORY_PATH.decode()}/rotation.csv')
# translation_file = open(f'{OUTPUT_DIRECTORY_PATH.decode()}.translation.csv')

# Start drone.
tello = Tello()
tello.connect()
tello.takeoff()
tello.streamon()
frame_read = tello.get_frame_read()

# Main loop.
rigid_body = ctypes.c_int(0)
motive.TT_RigidBodyResetOrientation(rigid_body)
frame = frame_read.frame
cv2.imshow('Tello Stream', frame)
key = cv2.waitKey(1) & 0xff
while key != ord('q'):
    frame = frame_read.frame
    x, y, z = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
    qx, qy, qz, qw = ctypes.c_float(), ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
    pitch, yaw, roll = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
    motive.TT_RigidBodyLocation(rigid_body, ctypes.byref(x), ctypes.byref(y), ctypes.byref(z), ctypes.byref(qx),
                                ctypes.byref(qy), ctypes.byref(qz), ctypes.byref(qw), ctypes.byref(pitch),
                                ctypes.byref(yaw), ctypes.byref(roll))
    print(f'Position:', (x.value, y.value, z.value))
    print(f'Orientation:', (pitch.value, yaw.value, roll.value), (qx.value, qy.value, qz.value, qw.value))
    cv2.imshow('Tello Stream', frame)
    key = cv2.waitKey(33) & 0xff

# Close OptiTrack and everything else.
check_return_code(motive.TT_Shutdown())
rotation_file.close()
translation_file.close()

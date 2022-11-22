"""

This script rotates a Tello drone that is registered as a rigid body in Motive.
The drones video stream is saved to images and OptiTrack rotation and translation is saved to a csv file.

"""


import sys
from ctypes import c_int, c_char_p, c_float, byref, POINTER, cdll
from djitellopy import Tello
import cv2
from time import sleep


# Parameters.
PROJECT_PATH = sys.argv[1].encode()             # Path to the .ttp project.
OUTPUT_DIRECTORY_PATH = sys.argv[2].encode()    # Path to a directory to save the output to.
LIBRARY_PATH = sys.argv[3]                      # Path to the NPTrackingToolsx64.dll file.
print(f'{OUTPUT_DIRECTORY_PATH.decode()}/rotation.csv')

# Load the Motive API library.
print('Loading dll:', LIBRARY_PATH)
motive = cdll.LoadLibrary(LIBRARY_PATH)

# Setup library functions.
motive.TT_Initialize.argtype = tuple()
motive.TT_Initialize.restype = c_int
motive.TT_LoadProject.argtype = (c_char_p,)
motive.TT_Initialize.restype = c_int
motive.TT_CameraCount.argtype = tuple()
motive.TT_CameraCount.restype = c_int
motive.TT_CameraName.argtype = (c_int,)
motive.TT_CameraName.restype = c_char_p
motive.TT_CameraFrameRate.argtype = (c_int,)
motive.TT_CameraFrameRate.restype = c_int
motive.TT_CameraExposure.argtype = (c_int,)
motive.TT_CameraExposure.restype = c_int
motive.TT_CameraIntensity.argtype = (c_int,)
motive.TT_CameraIntensity.restype = c_int
motive.TT_CameraVideoType.argtype = (c_int,)
motive.TT_CameraVideoType.argtype = c_int
motive.TT_RigidBodyCount.argtype = tuple()
motive.TT_RigidBodyCount.restype = c_int
motive.TT_RigidBodyName.argtype = (c_int,)
motive.TT_RigidBodyName.restype = c_char_p
motive.TT_RigidBodyLocation.argtype = (c_int, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
                                       POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
                                       POINTER(c_float), POINTER(c_float))
motive.TT_RigidBodyLocation.restype = None
motive.TT_RigidBodyResetOrientation.argtype = (c_int,)
motive.TT_RigidBodyResetOrientation.restype = c_int
motive.TT_IsRigidBodyTracked.argtype = (c_int,)
motive.TT_IsRigidBodyTracked.restype = c_int


def check_return_code(return_code: int, exit_on_error: bool = True):
    """
    Handle the Motive APIs return codes.

    :param return_code: The NPRESULT code that was returned in an operation.
    :param exit_on_error: Exit is error code is not 0.
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
    if exit_on_error:
        while motive.TT_Shutdown() != 0:
            print('Trying to shutdown...')
        print('Done.')


# Initialize the OptiTrack system.
print('Initializing OptiTrack...')
check_return_code(motive.TT_Initialize())

# Load the .ttp project.
print('Loading project:', PROJECT_PATH.decode())
check_return_code(motive.TT_LoadProject(c_char_p(PROJECT_PATH)))

# List detected cameras.
print('Cameras:')
for i in range(motive.TT_CameraCount()):
    print(f'    Camera #{i}:')
    i = c_int(i)
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
    i = c_int(i)
    print(f'        Name:', motive.TT_RigidBodyName(i).decode())
    x, y, z = c_float(), c_float(), c_float()
    qx, qy, qz, qw = c_float(), c_float(), c_float(), c_float()
    pitch, yaw, roll = c_float(), c_float(), c_float()
    motive.TT_RigidBodyLocation(i, byref(x), byref(y), byref(z), byref(qx),
                                byref(qy), byref(qz), byref(qw), byref(pitch),
                                byref(yaw), byref(roll))
    print(f'        Position:', (x.value, y.value, z.value))
    print(f'        Orientation:', (pitch.value, yaw.value, roll.value), (qx.value, qy.value, qz.value, qw.value))

# Files to keep the OptiTrack's data.
rotation_file = open(f'{OUTPUT_DIRECTORY_PATH.decode()}/rotation.csv', 'w+')
translation_file = open(f'{OUTPUT_DIRECTORY_PATH.decode()}/translation.csv', 'w+')

# Start drone.
print('Connecting to the drone and taking off...')
tello = Tello()
tello.connect()
tello.takeoff()
tello.streamon()
frame_read = tello.get_frame_read()

# Main loop.
print('Starting main loop...')
print('Tracking rigid body #0.')
drone_rigid_body_index = c_int(0)
key = 0
i = 0
motive.TT_RigidBodyResetOrientation(drone_rigid_body_index)
while key != ord('q'):
    tello.send_rc_control(0, 0, 0, 15)
    motive.TT_Update()
    if not motive.TT_IsRigidBodyTracked(drone_rigid_body_index):
        print('Drone not tracked.')
        key = cv2.waitKey(33) & 0xff
        continue
    frame = frame_read.frame
    x, y, z = c_float(), c_float(), c_float()
    qx, qy, qz, qw = c_float(), c_float(), c_float(), c_float()
    pitch, yaw, roll = c_float(), c_float(), c_float()
    motive.TT_RigidBodyLocation(drone_rigid_body_index, byref(x), byref(y), byref(z), byref(qx), byref(qy), byref(qz),
                                byref(qw), byref(pitch), byref(yaw), byref(roll))
    print(f'Position:', (x.value, y.value, z.value))
    print(f'Orientation:', (pitch.value, yaw.value, roll.value), (qx.value, qy.value, qz.value, qw.value))
    cv2.imwrite(f'{i}.png', frame)
    translation_file.write(f'{x},{y},{z}\n')
    rotation_file.write(f'{pitch},{yaw},{roll}\n')
    cv2.imshow('Tello Stream', frame)
    key = cv2.waitKey(33) & 0xff
    i += 1

# Close files and OptiTrack.
rotation_file.close()
translation_file.close()
check_return_code(motive.TT_Shutdown())

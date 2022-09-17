# ===================================================== IMPORTS ===================================================== #


import threading
from PIL import Image
from random import randint
import numpy as np
import cv2
try:
    from djitellopy import Tello
    import open3d as o3d
except (ImportError, Exception) as e:
    print('Failed to import some modules.')


# ===================================================== UTILS ======================================================= #


def openCV_projection(camera_matrix, rotation_matrix, camera_position, width, height, use_opencv: bool = True):
    pass

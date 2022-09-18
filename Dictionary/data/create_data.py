# ===================================================== IMPORTS ===================================================== #


import threading
from PIL import Image
from random import randint
import numpy as np
import cv2
import open3d as o3d

#TODO
# ===================================================== UTILS ======================================================= #


def project(points: np.ndarray, colors: np.ndarray, camera_matrix: np.ndarray, rotation_matrix: np.ndarray,
            camera_position: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Projects a 3D points to a 2D image.

    :param points: A numpy array of 3D points.
    :param colors: A numpy array of RGB triples.
    :param camera_matrix: The intrinsic camera matrix.
    :param rotation_matrix: The rotation matrix of the camera.
    :param camera_position: The position of the camera in the world.
    :param width: The width of the resulting image.
    :param height: The height of the resulting image.
    :return: An image as a numpy array.
    """
    projected_points, _ = cv2.projectPoints(points, rotation_matrix, camera_position, camera_matrix, distCoeffs=0,
                                            aspectRatio=(width / height))
    xs = np.round(projected_points[:, 0, 0]).astype(int)
    ys = np.round(projected_points[:, 0, 1]).astype(int)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.zeros((height, width), dtype=np.uint8)
    for i in np.intersect1d(np.where((xs >= 0) & (xs < width))[0],
                            np.where((ys >= 0) & (ys < height))[0], assume_unique=True):
        x, y, z = xs[i], ys[i], points[i, 2]
        if z > z_buffer[y, x]:
            image[y, x] = colors[i]
            z_buffer[y, x] = z
    return image


def open3D_manual_projection():
    pass


def main():
    camera_position = np.array([0, 0, 0], np.float32)
    fov_x, fov_y = 60, 60
    w, h = 1000, 1000
    cx, cy = w / 2, h / 2
    fx, fy = w / (2 * np.tan(np.deg2rad(fov_x))), h / (2 * np.tan(np.deg2rad(fov_y)))
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], np.float64)
    aspect_ratio = w / h
    sphere = create_sphere(read_from='sphere(180x180-0.025x0.025).pcd')
    points = np.asarray(sphere.points)
    colors = []
    for _ in range(len(points) // 100):
        rgb = np.transpose([randint(0, 255), randint(0, 255), randint(0, 255)])
        colors += [
                            rgb
                        ] * 100
    colors = np.asarray(colors)
    j = 0
    for alpha in np.arange(0, 1.51, 0.05):
        rad = np.deg2rad(alpha)
        sin, cos = np.sin(rad), np.cos(rad)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos, -sin],
            [0, sin, cos]
        ])
        cv2.imwrite(f'synthetic/6/{j}.png',project(
            points, colors, camera_matrix, rotation_matrix, camera_position, 1000, 1000
        ))
        j += 1




def create_sphere(longitude_step: float = 1, latitude_step: float = 1, longitude: int = 360, latitude: int = 360,
                  save_to: str = None, read_from: str = None, map_image: str = None):
    if read_from is not None:
        return o3d.io.read_point_cloud(read_from)
    sphere_array = []
    for i in np.arange(0, latitude, latitude_step):
        for j in np.arange(0, latitude, latitude_step):
            rad_i, rad_j = np.deg2rad(i), np.deg2rad(j)
            cos_i = np.cos(rad_i)
            sphere_array.append([cos_i * np.sin(rad_j),
                                 cos_i * np.cos(rad_j),
                                 np.sin(rad_i)])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sphere_array)
    if map_image is not None:
        colors = []
        for _ in range(len(sphere_array) // 50):
            rgb = np.transpose([randint(0, 255) / 255, randint(0, 255) / 255, randint(0, 255) / 255])
            colors += [rgb, rgb, rgb, rgb, rgb, rgb, rgb, rgb, rgb, rgb,
                       rgb, rgb, rgb, rgb, rgb, rgb, rgb, rgb, rgb, rgb,
                       rgb, rgb, rgb, rgb, rgb, rgb, rgb, rgb, rgb, rgb,
                       rgb, rgb, rgb, rgb, rgb, rgb, rgb, rgb, rgb, rgb,
                       rgb, rgb, rgb, rgb, rgb, rgb, rgb, rgb, rgb, rgb]
        # image = cv2.imread(map_image) / 255
        # center = image.shape[1] // 2, image.shape[0] // 2
        # for lat in np.arange(0, latitude, latitude_step):
        #     length = 0
        #     while length < center[0]:
        #         final_x = center[0] + (length * np.cos(np.deg2rad(lat)))
        #         final_y = center[1] - (length * np.sin(np.deg2rad(lat)))
        #         length += 1
        #         colors.append(
        #             np.transpose(image[int(final_y), int(final_x)])
        #         )
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if save_to is not None:
        o3d.io.write_point_cloud(save_to, pcd)
    return pcd





if __name__ == '__main__':
    main()

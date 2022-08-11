import numpy as np
import open3d as o3d
import cv2
from copy import deepcopy
from typing import Tuple


# ======================================= create or read point-clouds ======================================= #


def read_ply_file(file_path: str, color: Tuple[int, int, int] = None):
    """
    Read a .ply file using open3D.

    :param file_path: The path to the .ply file.
    :param color: Paint the 3D object with a uniform color (optional).
    :return: A geometry::PointCloud object for point clouds and geometry::TriangleMesh object for meshes.
    """
    geometry = o3d.io.read_point_cloud(file_path)
    if color is not None:
        geometry.paint_uniform_color(color)
    return geometry


def create_sphere(latitude: int = 360, longitude: int = 360, latitude_step: int = 1, longitude_step: int = 1,
                  color: Tuple[int, int, int] = None) -> np.ndarray:
    """
    Creates a sphere point cloud.

    :param latitude: The latitude of the sphere.
    :param longitude: The longitude of the sphere.
    :param latitude_step: The step in the latitude.
    :param longitude_step: The step in the longitude.
    :param color: The uniform color of the sphere (optional).
    :return: A PointCloud object.
    """
    sphere_points = []
    i, j = 0, 0
    while i < latitude:
        j = 0
        latitude_radians = np.deg2rad(i)
        while j < longitude:
            longitude_radians = np.deg2rad(j)
            sphere_points.append(
                [
                    np.cos(latitude_radians) * np.sin(longitude_radians),
                    np.cos(latitude_radians) * np.cos(longitude_radians),
                    np.sin(latitude_radians)
                ]
            )
            j += longitude_step
        i += latitude_step
    sphere_point_cloud = o3d.geometry.PointCloud()
    sphere_point_cloud.points = o3d.utility.Vector3dVector(sphere_points)
    if color is None:
        sphere_point_cloud.colors = o3d.utility.Vector3dVector(sphere_points)
    else:
        sphere_point_cloud.colors = o3d.utility.Vector3dVector([color] * len(sphere_points))
    return sphere_point_cloud


def mesh_to_point_cloud(mesh: o3d.geometry.TriangleMesh, numer_of_points: int):
    """
    Transform mesh to point cloud.

    :param mesh: A TriangleMesh object.
    :param numer_of_points: The density of points.
    :return: A point cloud object.
    """
    return o3d.geometry.sample_points_uniformly(deepcopy(mesh), number_of_points=numer_of_points)


# ======================================= transform point-clouds ======================================= #


def translate_point_cloud(point_cloud: o3d.geometry.PointCloud,
                          translation: Tuple[int, int, int]) -> o3d.geometry.PointCloud:
    """
    Translate a PointCloud object.

    :param point_cloud: The point-cloud object to transform.
    :param translation: The translation to perform.
    :return: A transformed copy of the geometry.
    """
    return deepcopy(point_cloud).translate(translation)


def rotate_point_cloud(point_cloud: o3d.geometry.PointCloud, rotation_matrix: np.ndarray,
                       center: Tuple[int, int, int] = (0, 0, 0)) -> o3d.geometry.PointCloud:
    """
        Rotate a PointCloud object.

        :param point_cloud: The point-cloud object to transform.
        :param rotation_matrix: The rotation matrix to use.
        :param center: The center point of the rotation.
        :return: A transformed copy of the geometry.
    """
    return deepcopy(point_cloud).rotate(rotation_matrix, center=center)


def scale_point_cloud(point_cloud: o3d.geometry.PointCloud, scalar: float):
    """
        Scale a PointCloud object.

        :param point_cloud: The point-cloud object to transform.
        :param scalar: The scalar to scale with.
        :return: A transformed copy of the geometry.
    """
    return deepcopy(point_cloud).scale(scalar)


def get_rotation_matrix(x_rotation: float, y_rotation: float, z_rotation: float) -> np.ndarray:
    theta_x = np.deg2rad(x_rotation)
    theta_y = np.deg2rad(y_rotation)
    theta_z = np.deg2rad(z_rotation)
    x_mat = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    y_mat = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    z_mat = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    return np.matmul(x_mat, y_mat, z_mat)


# ======================================= view or project geometries ======================================= #


def view_point_clouds(*point_clouds: o3d.geometry.PointCloud) -> None:
    """
    View one or more PointCloud objects.

    :param point_clouds: A PointCloud object.
    """
    o3d.visualization.draw_geometries_with_custom_animation(point_clouds)


def project_point_cloud(resolution: Tuple[int, int], eye_location: Tuple[int, int, int], camera_matrix: np.ndarray,
                        camera_rotation_matrix, hom_matrix: np.ndarray,
                        point_cloud: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Project a point-cloud.
    """
    image = np.full((resolution[1], resolution[0], 3), 255, dtype=np.uint8)
    z_buffer = np.zeros((resolution[1], resolution[0]), np.uint8)
    image_points = np.empty_like(point_cloud.points)
    T = np.column_stack([camera_rotation_matrix, eye_location])
    T = np.row_stack([T, [0, 0, 0, 1]])
    C = np.matmul(camera_matrix, hom_matrix)
    T = np.matmul(C, T)
    for idx, i in enumerate(np.asarray(point_cloud.points)):
        x = np.append(i, 1)
        x = np.matmul(T, x)
        x = np.transpose(x)
        if x[2] == 0:
            image_points[idx][0] = 0
            image_points[idx][1] = 0
        else:
            image_points[idx][0] = x[0] / x[2]
            image_points[idx][1] = x[1] / x[2]
        image_points[idx][2] = x[2]
    tX = image_points[:, 0]
    tY = image_points[:, 1]
    tZ = image_points[:, 2]
    xs = np.round(tX)
    ys = np.round(tY)
    xs = xs.astype(int)
    ys = ys.astype(int)
    point_color = np.asarray(point_cloud.colors)
    for idx, i in enumerate(xs):
        if 0 <= i < len(image[:, ]) and 0 <= ys[idx] < len(image[0]):
            inv_norm_color = abs(point_color[idx] * 255).astype(int)
            if tZ[idx] > z_buffer[i, ys[idx]]:
                image[i, ys[idx]] = inv_norm_color
                cy, cx = i, ys[idx]
                s = 1
                p1 = (cx, cy)
                p2 = (cx + s, cy)
                p3 = (cx, cy + s)
                p4 = (cx + s, cy + s)
                p5 = (cx - s, cy)
                p6 = (cx, cy - s)
                p7 = (cx - s, cy - s)
                p8 = (cx + s, cy - s)
                p9 = (cx - s, cy + s)
                for px, py in (p1, p2, p3, p4, p5, p6, p7, p8, p9):
                    px, py = max(px, 0), max(py, 0)
                    px, py = min(resolution[0] - 1, px), min(resolution[1] - 1, py)
                    image[py, px] = inv_norm_color
                z_buffer[i, ys[idx]] = tZ[idx]
    return image


def calculate_rotated_points(focal_length: float, rotation: float, points):
    rads = np.deg2rad(rotation)
    cos, sin = np.cos(rads), np.sin(rads)
    new_points = []
    for x, y, z in points:
        rev_proj_mat = np.array(
            # [
            #     [cos / z, 0, focal_length * sin],
            #     [0, 1 / z, 0],
            #     [-sin / (2 * focal_length), 0, cos]
            # ]
            [
                [1 / focal_length, 0, 0],
                [0, 1 / focal_length, 0],
                [0, 0, z]
            ]
        )
        new_points.append(np.matmul(rev_proj_mat, np.array([x, y, 1])))
    return np.array(new_points)


# ================================================== main ================================================== #


if __name__ == '__main__':

    # CAMERA SETTINGS
    _resolution = (480, 480)
    _theta = 0
    _eye_location = (0, 0, 0)
    _fov_x = 60
    _fov_y = 40
    _camera_matrix = np.array([
        [_resolution[0] / (2 * np.tan(np.deg2rad(_fov_x))), 0, _resolution[0] / 2],
        [0, _resolution[1] / (2 * np.tan(np.deg2rad(_fov_y))), _resolution[1] / 2],
        [0, 0, 1]
    ])

    # READ/CREATE POINT-CLOUD
    sphere_object = create_sphere(360, 360, 1, 1)

    # ROTATE THE CAMERA
    images = []
    for _theta in range(30, 30, 10):

        # CREATE RELEVANT MATRICES
        hom_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        rot_mat = get_rotation_matrix(90, 50, 180 + _theta)

        # PROJECT THE OBJECT ONTO AN IMAGE
        image = project_point_cloud(_resolution, _eye_location, _camera_matrix, rot_mat, hom_mat, sphere_object)
        images.append(image)

        # SHOW THE IMAGE
        # cv2.imshow('', image)
        # cv2.waitKey()

        # SAVE THE IMAGE TO A FILE
        pass

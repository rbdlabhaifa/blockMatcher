from random import randint
import numpy as np
import open3d as o3d
import cv2
from copy import deepcopy
from typing import Tuple


# ======================================= create or read geometries ======================================= #


def read_ply_file(file_path: str, read_as: str = 'mesh', color: Tuple[int, int, int] = None) -> o3d.Geometry:
    """
    Read a .ply file using open3D.

    :param file_path: The path to the .ply file.
    :param read_as: Read the file as a 'mesh' or as a 'point cloud'.
    :param color: Paint the 3D object with a uniform color (optional).
    :return: A geometry::PointCloud object for point clouds and geometry::TriangleMesh object for meshes.
    """
    if read_as == 'mesh':
        geometry = o3d.io.read_triangle_mesh(file_path)
        if color is not None:
            geometry.paint_uniform_color(color)
        return geometry
    elif read_as == 'point cloud':
        geometry = o3d.io.read_point_cloud(file_path)
        if color is not None:
            geometry.paint_uniform_color(color)
        return geometry
    else:
        raise ValueError(f'parameter \'read_as\' can only be \'mesh\' or \'point cloud\', not {read_as}.')


def create_box(width: int, height: int, depth: int, color: Tuple[int, int, int] = None) -> o3d.Geometry:
    """
    Creates a box.

    :param width: The width of the box.
    :param height: The height of the box.
    :param depth: The depth of the box.
    :param color: The color of the box.
    :return: A mesh of a box.
    """
    mesh = o3d.geometry.create_mesh_box(width, height, depth)
    if color is not None:
        mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def create_sphere(radius: int, color: Tuple[int, int, int] = None) -> o3d.Geometry:
    """
    Creates a sphere.

    :param radius: The radius of the sphere.
    :param color: The color of the sphere.
    :return: A mesh of a sphere.
    """
    mesh = o3d.geometry.create_mesh_sphere(radius)
    if color is not None:
        mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def create_sphere_point_cloud(radius: int = 360, degree_step: int = 1):
    """
    Create sphere point cloud.

    :param radius: Radius of the sphere.
    :param degree_step: The step in degrees. For the density of points.
    :return: A list of points as an open3d.utility.Vector3dVector object.
    """
    deg_step_1 = degree_step
    deg_step_2 = degree_step
    latitude = radius
    longitude = radius
    sphere_array = []
    for i in range(0, latitude, deg_step_1):
        for j in range(0, longitude, deg_step_2):
            x = np.array([np.cos(np.deg2rad(i)) * np.sin(np.deg2rad(j)),
                          np.cos(np.deg2rad(i)) * np.cos(np.deg2rad(j)),
                          np.sin(np.deg2rad(i))])
            sphere_array.append(x)
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(sphere_array)
    return pcl


def create_cylinder(radius: int, height: int, color: Tuple[int, int, int] = None) -> o3d.Geometry:
    """
    Creates a cylinder.

    :param radius: The radius of the cylinder.
    :param height: the height of the cylinder.
    :param color: The color of the cylinder.
    :return: A mesh of a cylinder.
    """
    mesh = o3d.geometry.create_mesh_cylinder(radius, height)
    if color is not None:
        mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def mesh_to_point_cloud(mesh: o3d.geometry.TriangleMesh, numer_of_points: int):
    """
    Transform mesh to point cloud.

    :param mesh: A TriangleMesh object.
    :param numer_of_points: The density of points.
    :return: A point cloud object.
    """
    return o3d.geometry.sample_points_uniformly(deepcopy(mesh), number_of_points=numer_of_points)


# ======================================= transform geometries ======================================= #


def translate_geometry(geometry: o3d.Geometry, translation: Tuple[int, int, int]):
    """
    Translate a Geometry object.

    :param geometry: The geometry object to transform.
    :param translation: The translation to perform.
    :return: A transformed copy of the geometry.
    """
    return deepcopy(geometry).translate(translation)


def rotate_geometry(geometry: o3d.Geometry, rotation_matrix: np.ndarray, center: Tuple[int, int, int] = (0, 0, 0)):
    """
        Rotate a Geometry object.

        :param geometry: The geometry object to transform.
        :param rotation_matrix: The rotation matrix to use.
        :param center: The center point of the rotation.
        :return: A transformed copy of the geometry.
    """
    return deepcopy(geometry).rotate(rotation_matrix, center=center)


def scale_geometry(geometry: o3d.Geometry, scalar: float):
    """
        Scale a Geometry object.

        :param geometry: The geometry object to transform.
        :param scalar: The scalar to scale with.
        :return: A transformed copy of the geometry.
    """
    return deepcopy(geometry).scale(scalar)


# ======================================= view or project geometries ======================================= #


def view_geometries(*geometries: o3d.Geometry) -> None:
    """
    View one or more Geometry objects.

    :param geometries: A Geometry object.
    """
    o3d.visualization.draw_geometries_with_custom_animation(geometries)


def project_point_clouds(resolution: Tuple[int, int], theta: float, eye_location: Tuple[int, int, int],
                         camera_matrix: np.ndarray, *clouds: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Projects geometries onto an image using openCV.

    :param resolution: The resolution of the image.
    :param theta: The rotation of the camera.
    :param eye_location: The coordinates of the camera in 3D space.
    :param camera_matrix: The transformation defined by the camera (?).
    :param clouds: A list of geometries objects to project.
    :return:
    """
    points = np.array([p for geometry in clouds for p in np.asarray(geometry.points)])
    colors = np.array([p for geometry in clouds for p in np.asarray(geometry.colors)])
    radians = np.deg2rad(theta)
    view_plane = np.zeros((resolution[1], resolution[0], 3))
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians), 0],
        [np.sin(radians), np.cos(radians), 0],
        [0, 0, 1]
    ])
    projected_points = cv2.projectPoints(points, rotation_matrix, eye_location, camera_matrix, distCoeffs=0,
                                         aspectRatio=(resolution[0] / resolution[1]))
    projected_points = np.asarray(projected_points[0])
    xs = np.round([x[0][0] for x in projected_points]).astype(int)
    ys = np.round([x[0][1] for x in projected_points]).astype(int)
    for idx, i in enumerate(xs):
        if 0 <= i < len(view_plane[:, ]) and 0 <= ys[idx] < len(view_plane[0]):
            inv_norm_color = (colors[idx] * 255).astype(int)
            view_plane[i, ys[idx]] = inv_norm_color
    return view_plane


# ================================================== main ================================================== #


if __name__ == '__main__':

    # NOTE: SOME THINGS WORK ON POINT CLOUDS AND NOT ON MESHES, THE OPPOSITE IS TRUE AS WELL.

    # CAMERA SETTINGS
    _resolution = (500, 500)
    _theta = 0
    _eye_location = (0, 0, -40)
    _fov_x = 60
    _fov_y = 40
    _camera_matrix = np.array([
        [_resolution[0] / (2 * np.tan(np.deg2rad(_fov_x))), 0, _resolution[0] / 2],
        [0, _resolution[1] / (2 * np.tan(np.deg2rad(_fov_y))), _resolution[1] / 2],
        [0, 0, 1]
    ])

    # READ/CREATE GEOMETRY OBJECT
    geometry_object = create_sphere_point_cloud()

    # ROTATE THE CAMERA
    for _theta in range(30, 2000, 10):
        # PROJECT THE OBJECT ONTO AN IMAGE
        image = project_point_clouds(_resolution, _theta, _eye_location, _camera_matrix, geometry_object)

        # SHOW THE IMAGE
        cv2.imshow('', image)
        cv2.waitKey()

        # SAVE THE IMAGE TO A FILE
        pass

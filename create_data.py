# ===================================================== IMPORTS ===================================================== #


from typing import Tuple, List
from random import randint
import numpy as np
import cv2
try:
    import open3d as o3d
except ImportError:
    print('Could not import open3D.')
try:
    from djitello import Tello
except ImportError:
    print('Could not import djitello.')


# ===================================================== PROJECTION ================================================== #


def create_sphere(latitude: int = 360, longitude: int = 360, step: Tuple[float, float] = (1, 1),
                  save_to: str = None) -> o3d.geometry.PointCloud:
    """
    Creates a PointCloud sphere.

    :param latitude: The latitude of the sphere.
    :param longitude: The longitude of the sphere.
    :param step: The step in latitude and longitude.
    :param save_to: The file to save the sphere to (has to end with .pcd), set to None to not save.
    :return: The sphere.
    """
    sphere_array = []
    for i in np.arange(0, latitude, step[0]):
        for j in np.arange(0, longitude, step[1]):
            rad_i, rad_j = np.deg2rad(i), np.deg2rad(j)
            cos_i = np.cos(rad_i)
            sphere_array.append([cos_i * np.sin(rad_j),
                                 cos_i * np.cos(rad_j),
                                 np.sin(rad_i)])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sphere_array)
    if save_to is not None:
        o3d.io.write_point_cloud(save_to, pcd)
    return pcd


def load_sphere(pcd_file_path: str) -> o3d.geometry.PointCloud:
    """
    Creates a PointCloud sphere from a pcd file.

    :param pcd_file_path: The path to a pcd file of a sphere.
    :return: The sphere.
    """
    return o3d.io.read_point_cloud(pcd_file_path)


def color_sphere(sphere: o3d.geometry.PointCloud = None, point_count: int = None, similar: int = 100) -> np.ndarray:
    """
    Colors a sphere.

    :param sphere: The sphere to save the colors to, can be None as long as points_count is not None.
    :param point_count: The count of points in the sphere, can be None as long as sphere is not None.
    :param similar: The amount of points with similar colors.
    :return: The numpy array of colors.
    """
    if point_count is None:
        point_count = len(np.asarray(sphere.points))
    colors = []
    for _ in range(point_count // similar):
        rgb = np.transpose([randint(0, 255), randint(0, 255), randint(0, 255)])
        colors += [rgb] * similar
    colors = np.array(colors)
    if sphere is not None:
        sphere.colors = o3d.utility.Vector3dVector(colors)
    return colors


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


def project_manually(geometries: List[o3d.geometry.Geometry]):

    images_written = 0
    total_rot = 0
    folder = 4

    def load_render_optionX(vis):
        op = vis.get_render_option()
        op.mesh_color_option = o3d.visualization.MeshColorOption.XCoordinate
        op.mesh_show_back_face = True
        vis.update_renderer()
        return True

    def load_render_optionY(vis):
        op = vis.get_render_option()
        op.mesh_color_option = o3d.visualization.MeshColorOption.YCoordinate
        op.mesh_show_back_face = True
        vis.update_renderer()
        return True

    def load_render_optionZ(vis):
        op = vis.get_render_option()
        op.mesh_color_option = o3d.visualization.MeshColorOption.ZCoordinate
        op.mesh_show_back_face = True
        vis.update_renderer()
        return True

    def capture_image(vis):
        nonlocal images_written
        nonlocal folder
        vis.capture_screen_image(f'synthetic/{folder}/{images_written}.png')
        print(f'written frame {images_written}')
        images_written += 1
        return True

    def reset(vis):
        nonlocal total_rot
        nonlocal images_written
        nonlocal folder
        folder += 1
        total_rot = 0
        images_written = 0
        op = vis.get_render_option()
        op.point_size = 1.
        vc = vis.get_view_control()
        print('fov=', vc.get_field_of_view())
        cam = vc.convert_to_pinhole_camera_parameters()
        intrinsic = cam.intrinsic
        print('width=', intrinsic.width)
        print('height=', intrinsic.height)
        fx, fy = intrinsic.get_focal_length()
        cx, cy = intrinsic.get_principal_point()
        print('focal length=', (fx, fy))
        print('principal point=', (cx, cy))
        new_cam = o3d.camera.PinholeCameraParameters()
        new_cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsic.width, intrinsic.height, fx, fy, cx, cy)
        print('intrinsic=', intrinsic.intrinsic_matrix)
        new_cam.extrinsic = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        )
        print('extrinsic=', new_cam.extrinsic)
        vc.convert_from_pinhole_camera_parameters(new_cam, True)
        return True

    def rotate_to_the_right(vis):
        nonlocal total_rot
        total_rot += 0.1
        vc = vis.get_view_control()
        cam = vc.convert_to_pinhole_camera_parameters()
        width, height = cam.intrinsic.width, cam.intrinsic.height
        fx, fy = cam.intrinsic.get_focal_length()
        cx, cy = cam.intrinsic.get_principal_point()
        new_cam = o3d.camera.PinholeCameraParameters()
        new_cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        sin = np.sin(np.deg2rad(total_rot))
        cos = np.cos(np.deg2rad(total_rot))
        new_cam.extrinsic = np.array([
            [cos, 0, sin, 0],
            [0, 1, 0, 0],
            [-sin, 0, cos, 0],
            [0, 0, 0, 1]
        ])
        if vc.convert_from_pinhole_camera_parameters(new_cam, True):
            print('worked!')
        else:
            print('failed!')
            exit(1)
        print('rotated right, total rot is', total_rot)
        return True

    key_to_callback = {
        ord("Q"): load_render_optionX,
        ord("W"): load_render_optionY,
        ord("R"): load_render_optionZ,
        ord("D"): rotate_to_the_right,
        ord("S"): capture_image,
        ord("`"): reset,
    }

    print('press QWR for different gradients.')
    print('press D to rotate right and left respectively.')
    print('press S to capture an image.')
    print('press ` to change point size.')

    o3d.visualization.draw_geometries_with_key_callbacks(geometries, key_to_callback, width=1000, height=1000)


# ===================================================== DRONE ======================================================= #


def tello_drone(step: int = 20, image_folder: str = None):
    print('You can now control the drone freely with your keyboard.')
    print('Keys: w, a, s, d, e, q, r, f.')
    print('Press esc to stop and space to capture image.')
    tello = Tello()
    tello.connect()
    tello.takeoff()
    tello.streamon()
    image_number = 0
    total_rotation = 0
    while True:
        cv2.imshow('', tello.get_frame_read().frame)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break
        elif key == ord(' '):
            cv2.imwrite(f'{image_folder}/{image_number}.png', tello.get_frame_read().frame)
            image_number += 1
            print(f'saved image to {image_folder}/{image_number}.png')
        elif key == ord('w'):
            tello.move_forward(step)
            print('moved forward by', step)
        elif key == ord('s'):
            tello.move_back(step)
            print('moved backwards by', step)
        elif key == ord('a'):
            tello.move_left(step)
            print('moved left by', step)
        elif key == ord('d'):
            tello.move_right(step)
            print('moved right by', step)
        elif key == ord('e'):
            tello.rotate_clockwise(step)
            print('rotated clockwise by', step)
            total_rotation -= step
            print('total rotation is', total_rotation)
        elif key == ord('q'):
            tello.rotate_counter_clockwise(step)
            print('rotated counter-clockwise by', step)
            total_rotation += step
            print('total rotation is', total_rotation)
        elif key == ord('r'):
            tello.move_up(step)
            print('moved up by', step)
        elif key == ord('f'):
            tello.move_down(step)
            print('moved down by', step)
    tello.streamoff()
    tello.land()


# ===================================================== MAIN ======================================================== #


if __name__ == '__main__':
    pass

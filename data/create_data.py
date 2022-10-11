# ===================================================== IMPORTS ===================================================== #


from typing import Tuple, List
from random import randint
import numpy as np
import cv2
import threading
import open3d as o3d


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
    latitude_radians = np.deg2rad(np.arange(0, latitude + step[0], step[0]))
    latitude_sin = np.sin(latitude_radians)
    latitude_cos = np.cos(latitude_radians)
    longitude_radians = np.deg2rad(np.arange(0, longitude + step[1], step[1]))
    longitude_sin = np.sin(longitude_radians)
    longitude_cos = np.cos(longitude_radians)
    sphere_array = np.zeros((latitude_radians.shape[0] * longitude_radians.shape[0], 3), dtype=float)
    long_shape = longitude_radians.shape[0]
    for i in range(latitude_radians.shape[0]):
        print(i)
        for j in range(long_shape):
            cos_i = latitude_cos[i]
            sphere_array[i * long_shape + j] = ([cos_i * longitude_sin[j], cos_i * longitude_cos[j], latitude_sin[i]])
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


def project_manually(geometries: List[o3d.geometry.Geometry], image_folder: str, window_width: int, window_height: int):

    images_written = 0
    total_angle = 0

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
        vis.capture_screen_image(f'{image_folder}/{images_written}.png')
        print(f'written frame {images_written}')
        images_written += 1
        return True

    def reset(vis):
        nonlocal total_angle
        nonlocal images_written
        total_angle = 0
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
        nonlocal total_angle
        total_angle += 0.1
        vc = vis.get_view_control()
        cam = vc.convert_to_pinhole_camera_parameters()
        width, height = cam.intrinsic.width, cam.intrinsic.height
        fx, fy = cam.intrinsic.get_focal_length()
        cx, cy = cam.intrinsic.get_principal_point()
        new_cam = o3d.camera.PinholeCameraParameters()
        new_cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        sin = np.sin(np.deg2rad(total_angle))
        cos = np.cos(np.deg2rad(total_angle))
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
        print('rotated right, total angle is', total_angle)
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
    print('press D to rotate right.')
    print('press S to capture an image.')
    print('press ` to change point size.')

    o3d.visualization.draw_geometries_with_key_callbacks(geometries, key_to_callback,
                                                         width=window_width, height=window_height)


# ===================================================== DRONE ======================================================= #


def tello_drone(step_pointer: List[int] = (20,), image_folder: str = None):
    try:
        from djitello import Tello
    except ImportError as error:
        print(error)
        exit(1)
    print('You can now control the drone freely with your keyboard.')
    print('Keys: w, a, s, d, e, q, r, f.')
    print('Press esc to stop and space to capture image.')
    tello = Tello()
    tello.connect()
    tello.takeoff()
    tello.streamon()
    image_number = 0
    total_angle = 0
    while True:
        cv2.imshow('', tello.get_frame_read().frame)
        key = cv2.waitKey(1) & 0xff
        step = step_pointer[0]
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
            total_angle -= step
            print('total rotation is', total_angle)
        elif key == ord('q'):
            tello.rotate_counter_clockwise(step)
            print('rotated counter-clockwise by', step)
            total_angle += step
            print('total rotation is', total_angle)
        elif key == ord('r'):
            tello.move_up(step)
            print('moved up by', step)
        elif key == ord('f'):
            tello.move_down(step)
            print('moved down by', step)
    tello.streamoff()
    tello.land()


# ===================================================== MAIN ======================================================== #


def main_drone():
    image_folder = ''
    step = [20]
    drone_control_thread = threading.Thread(target=lambda *args: tello_drone(step, image_folder))
    drone_control_thread.run()
    while True:
        try:
            user_input = input('enter step or break: ')
            if user_input == 'break':
                break
            step[0] = int(user_input)
        except (ValueError, Exception):
            continue


def main_open3D():
    image_folder = ''
    width, height = 1000, 1000
    sphere = create_sphere()  # or load_sphere()
    project_manually([sphere], image_folder, width, height)


def main_openCV():
    image_folder = 'synthetic/rotation - y'
    axis = 'y'  # rotation around x, y or z axes.
    sphere = load_sphere('sphere(180, 180, 0.025, 0.025).pcd')  # or create_sphere()
    width, height = 1000, 1000
    fov_x, fov_y = 60, 60
    cx, cy = width // 2, height // 2
    fx, fy = width / (2 * np.tan(fov_x)), height / (2 * np.tan(fov_y))
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    camera_position = np.array([0, 0, 0], dtype=float)
    color_sphere(sphere, similar=200)
    points, colors = np.asarray(sphere.points), np.asarray(sphere.colors)
    start_angle, end_angle, angle_step = 0, 1.5, 0.1
    for i, angle in enumerate(np.arange(start_angle, end_angle + angle_step, angle_step)):
        print(i, angle)
        rads = np.deg2rad(angle)
        sin, cos = np.sin(rads), np.cos(rads)
        if axis == 'x':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, cos, -sin],
                [0, sin, cos]
            ])
        elif axis == 'y':
            rotation_matrix = np.array([
                [cos, 0, sin],
                [0, 1, 0],
                [-sin, 0, cos]
            ])
        else:
            rotation_matrix = np.array([
                [cos, -sin, 0],
                [sin, cos, 0],
                [0, 0, 1]
            ])
        image = project(points, colors, camera_matrix, rotation_matrix, camera_position, width, height)
        cv2.imwrite(image_folder + f'/{i}.png', image)


if __name__ == '__main__':
    # main_drone()
    # main_open3D()
    main_openCV()
    pass

from random import randint
import numpy as np
import open3d as o3d
import cv2


# SCRIPT FOR CREATING GRADIENT IMAGES.


def manualProjection(point_cloud_in_numpy, rotation_mat, cam_mat, hom_mat, eye, size):
    image = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    zBuffer = np.zeros((size[1], size[0]), np.uint8)

    img_points = np.empty_like(point_cloud_in_numpy)
    T = np.column_stack([rotation_mat, eye])
    T = np.row_stack([T, [0, 0, 0, 1]])
    C = np.matmul(cam_mat, hom_mat)
    T = np.matmul(C, T)
    for idx, i in enumerate(point_cloud_in_numpy):
        x = np.append(i, 1)
        x = np.matmul(T, x)
        x = np.transpose(x)
        if x[2] == 0:
            img_points[idx][0] = 0
            img_points[idx][1] = 0
        else:
            img_points[idx][0] = (x[0] / x[2])
            img_points[idx][1] = (x[1] / x[2])
        img_points[idx][2] = (x[2])

    tX = img_points[:, 0]
    tY = img_points[:, 1]
    tZ = img_points[:, 2]
    xs = np.round(tX)
    ys = np.round(tY)
    xs = xs.astype(int)
    ys = ys.astype(int)

    for idx, i in enumerate(xs):
        if 0 <= i < size[0] and 0 <= ys[idx] < size[1]:
            inv_norm_color = abs(point_cloud_in_numpy[idx] * 255).astype(int)
            if tZ[idx] > zBuffer[i, ys[idx]]:
                image[i, ys[idx]] = inv_norm_color
                zBuffer[i, ys[idx]] = tZ[idx]
    return image


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


def main_manual_gradient():
    im_size = (480, 480)
    fov_x = 60
    fov_y = 40
    end_angle = 10
    step_size = 0.1
    eye_location = np.array([0., 0., 0.])
    hom_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    cam_matrix = np.array([[im_size[0] / (2 * np.tan(np.deg2rad(fov_x))), 0, im_size[0] / 2],
                           [0, im_size[1] / (2 * np.tan(np.deg2rad(fov_y))), im_size[1] / 2],
                           [0, 0, 1]])

    pcl = create_sphere(0.1, .1)
    point_cloud_in_np = np.asarray(pcl.points)

    theta_x = np.deg2rad(0)
    theta_y = np.deg2rad(90)
    theta_z = np.deg2rad(0)
    x_rot = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    y_rot = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    z_rot = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    starting_rotation_mat = np.matmul(x_rot, y_rot)
    starting_rotation_mat = np.matmul(z_rot, starting_rotation_mat)

    iii = 0
    for alpha in np.arange(0, end_angle, step_size):
        print('doing rotation by', alpha)
        radd = np.deg2rad(alpha)
        sin, cos = np.sin(radd), np.cos(radd)
        rot_mat = np.matmul(starting_rotation_mat, np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]))
        img = manualProjection(point_cloud_in_np, rot_mat, cam_matrix, hom_matrix, eye_location, im_size)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        cv2.imshow('', img)
        cv2.waitKey()
        # cv2.imwrite("gradient/2/image" + str(iii) + ".png", img)
        iii += 1


# OPEN3D SCRIPT FOR CREATING GRADIENT IMAGES MANUALLY.


def custom_draw_geometry_with_key_callback(geometries):

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
        if vc.convert_from_pinhole_camera_parameters(new_cam, True):
            print('worked!')
        else:
            print('failed!')
            exit(1)
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


def main_open3D_gradient():
    sphere = create_sphere(read_from='sphere(180x180-0.025x0.025).pcd')
    colors = []
    for _ in range(len(np.asarray(sphere.points)) // 100):
        rgb = np.transpose([randint(0, 255), randint(0, 255), randint(0, 255)]) / 255
        colors += [
           rgb
        ] * 100
    sphere.colors = o3d.utility.Vector3dVector(colors)
    geometries_list = [sphere]
    custom_draw_geometry_with_key_callback(geometries_list)


# MAIN


def main_open3D():

    renderer = o3d.visualization





    # Create the sphere
    sphere = create_sphere(read_from='sphere(180x180-0.025x0.025).pcd')
    points = np.asarray(sphere.points)
    colors = []
    for _ in range(len(points) // 100):
        rgb = np.transpose([randint(0, 255), randint(0, 255), randint(0, 255)])
        colors += [
           rgb
        ] * 100
    colors = np.array(colors, dtype=np.uint8)
    # Create projection parameters
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
    for alpha in range(0, 51, 1):
        rad = np.deg2rad(alpha / 10)
        sin, cos = np.sin(rad), np.cos(rad)
        rotation_matrix = np.array([
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1]
        ])



def main_openCV():
    # Create the sphere
    sphere = create_sphere(read_from='sphere(180x180-0.025x0.025).pcd')
    points = np.asarray(sphere.points)
    colors = []
    for _ in range(len(points) // 100):
        rgb = np.transpose([randint(0, 255), randint(0, 255), randint(0, 255)])
        colors += [
           rgb
        ] * 100
    colors = np.array(colors, dtype=np.uint8)
    # Create projection parameters
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
    for alpha in range(0, 16, 1):
        rad = np.deg2rad(alpha / 10)
        sin, cos = np.sin(rad), np.cos(rad)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos, -sin],
            [0, sin, cos]
        ])
        print('projecting...')
        image_points, _ = cv2.projectPoints(points, rotation_matrix, camera_position, camera_matrix,
                                            distCoeffs=0, aspectRatio=aspect_ratio)
        print('done.')
        image = np.zeros((h, w, 3), np.uint8)
        z_buffer = np.zeros((h, w))
        deleted = 0
        for i, p in enumerate(image_points):
            x, y = p[0]
            x, y = int(round(x)), int(round(y))
            z = points[i - deleted, 2]
            if 0 <= x < w and 0 <= y < h and z > z_buffer[y, x]:
                z_buffer[y, x] = z
                image[y, x] = colors[i - deleted]
        print('deleted:', deleted)
        print(f'Writing image {alpha}...')
        cv2.imwrite(f'C:/Users/BenGo/PycharmProjects/blockMatcher/Dictionary/data/synthetic/7/{alpha}.png', image)
        print('done.')


if __name__ == '__main__':
    # main_manual_gradient()
    # main_open3D_gradient()
     main_openCV()


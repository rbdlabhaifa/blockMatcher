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


def create_sphere(longitude_step: float = 1, latitude_step: float = 1, longitude: int = 360, latitude: int = 360):
    sphere_array = []
    i, j = 0, 0
    while i < latitude:
        j = 0
        while j < longitude:
            rad_i, rad_j = np.deg2rad(i), np.deg2rad(j)
            cos_i = np.cos(rad_i)
            sphere_array.append([cos_i * np.sin(rad_j),
                                 cos_i * np.cos(rad_j),
                                 np.sin(rad_i)])
            j += longitude_step
        i += latitude_step
    pcl1 = o3d.geometry.PointCloud()
    pcl1.points = o3d.utility.Vector3dVector(sphere_array)
    return pcl1


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
        vis.capture_screen_image(f'gradient/4/{images_written}.png')
        print(f'written frame {images_written}')
        images_written += 1
        return True

    def rotate_to_the_left(vis):
        vc = vis.get_view_control()
        vc.camera_local_rotate(-0.1, 0, 0)
        return True

    def rotate_to_the_right(vis):
        vc = vis.get_view_control()
        vc.camera_local_rotate(0.1, 0, 0)
        print('rotated!')
        return True

    key_to_callback = {
        ord("X"): load_render_optionX,
        ord("Y"): load_render_optionY,
        ord("Z"): load_render_optionZ,
        ord("-"): rotate_to_the_right,
        ord("="): rotate_to_the_left,
        ord("."): capture_image,
    }

    o3d.visualization.draw_geometries_with_key_callbacks(geometries, key_to_callback, width=480, height=480)


def main_open3D_gradient():
    sphere = o3d.geometry.TriangleMesh().create_sphere(1, 100)
    print(sphere)
    geometries_list = [sphere]
    custom_draw_geometry_with_key_callback(geometries_list)


# MAIN


if __name__ == '__main__':
     # main_manual_gradient()
    main_open3D_gradient()

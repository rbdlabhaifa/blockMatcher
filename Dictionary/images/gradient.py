import numpy as np
import open3d as o3d
import cv2


class points_cloud:
    def __init__(self, pcd, eye_location):
        self.pcd = pcd
        self.eye_location = eye_location

    def visualize(self):
        # Visualize the point cloud within open3d
        mesh_frame = o3d.geometry.TriangleMesh.create_mesh_coordinate_frame(size=6, origin=eye_location)
        o3d.visualization.draw_geometries_with_custom_animation([self.pcd, mesh_frame])

    def openCVprojection(self, alpha):
        img_points = np.empty_like(self.pcd.points)
        rad = np.deg2rad(alpha)
        image = np.zeros((size[1], size[0], 3), np.uint8)
        rotation_mat = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
        img_points = cv2.projectPoints(point_cloud_in_numpy, rotation_mat, eye_location, cam_mat, distCoeffs=0)
        img_points = np.asarray(img_points[0])
        xs = np.round([x[0][0] for x in img_points])
        ys = np.round([x[0][1] for x in img_points])
        xs = xs.astype(int)
        ys = ys.astype(int)

        for idx, i in enumerate(xs):
            if 0 <= i < len(image[:, ]) and 0 <= ys[idx] < len(image[0]):
                inv_norm_color = abs(point_color[idx] * 255).astype(int)
                image[i, ys[idx]] = inv_norm_color
                image[min(i + 1, image.shape[0] - 1), min(ys[idx] + 1, image.shape[1] - 1)] = inv_norm_color
                image[max(0, i - 1), max(ys[idx] - 1, 0)] = inv_norm_color
                image[i, min(ys[idx] + 1, image.shape[1]-1)] = inv_norm_color
                image[i, max(ys[idx] - 1, 0)] = inv_norm_color
                image[min(i + 1, image.shape[0] - 1), max(ys[idx] - 1, 0)] = inv_norm_color
                image[max(0, i - 1), min(ys[idx] + 1, image.shape[1] - 1)] = inv_norm_color
                image[min(i + 1, image.shape[0] - 1), ys[idx]] = inv_norm_color
                image[max(0, i - 1), ys[idx]] = inv_norm_color
        return image

    def manualProjection(self, alpha):
        rad = np.deg2rad(alpha)
        image = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
        zBuffer = np.zeros((size[1], size[0]), np.uint8)
        rotation_mat = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])


        theta_x = np.deg2rad(90)
        theta_y = np.deg2rad(50)
        theta_z = np.deg2rad(180)
        x_rot = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
        y_rot = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
        z_rot = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])

        rotation_mat = np.matmul(x_rot, rotation_mat)
        rotation_mat = np.matmul(y_rot, rotation_mat)
        rotation_mat = np.matmul(z_rot, rotation_mat)

        img_points = np.empty_like(self.pcd.points)
        T = np.column_stack([rotation_mat, eye_location])
        T = np.row_stack([T, [0, 0, 0, 1]])
        C = np.matmul(cam_mat, hom_matrix)
        T = np.matmul(C, T)
        for idx, i in enumerate(point_cloud_in_numpy):
            x = np.append(i, 1)
            x = np.matmul(T, x)
            x = np.transpose(x)
            if x[2] == 0:
                img_points[idx][0] = 0
                img_points[idx][1] = 0
            else:
                img_points[idx][0] = x[0] / x[2]
                img_points[idx][1] = x[1] / x[2]
            img_points[idx][2] = x[2]

        tX = img_points[:, 0]
        tY = img_points[:, 1]
        tZ = img_points[:, 2]
        xs = np.round(tX)
        ys = np.round(tY)
        xs = xs.astype(int)
        ys = ys.astype(int)

        l1, l2 = 10, 10
        w, h = 1000, 100
        square = {i * 360 + j for j in range(l2, l2 + w) for i in range(l1, l1 + h)}
        for idx, i in enumerate(xs):
            if 0 <= i < len(image[:, ]) and 0 <= ys[idx] < len(image[0]):
                inv_norm_color = abs(point_color[idx] * 255).astype(int)
                if idx in square:
                    inv_norm_color = (0, 0, 0)
                    print('yay')
                if tZ[idx] > zBuffer[i, ys[idx]]:
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
                        px, py = min(size[0] - 1, px), min(size[1] - 1, py)
                        image[py, px] = inv_norm_color
                    zBuffer[i, ys[idx]] = tZ[idx]

        return image


def create_sphere():
    sphere_array = []
    longitude = 360
    deg_step_1 = 1
    deg_step_2 = 1
    latitude = 360
    i, j = 0, 0
    while i < latitude:
        j = 0
        while j < longitude:
            x = np.array([np.cos(np.deg2rad(i)) * np.sin(np.deg2rad(j)),
                          np.cos(np.deg2rad(i)) * np.cos(np.deg2rad(j)),
                          np.sin(np.deg2rad(i))])
            sphere_array.append(x)
            j += deg_step_2
        i += deg_step_1
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(sphere_array)
    # o3d.visualization.draw_geometries([pcl])
    return pcl


if __name__ == '__main__':
    # Initialization
    img_array = []
    size = (480, 480)
    fov_x = 60
    fov_y = 40
    end_angle = 10
    step_size = 1
    eye_location = np.array([0., 0., 0.])
    hom_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    manualProjection = True
    sphere = True
    # Read .ply file
    input_file = "/home/txp1/Downloads/city-quay/base.ply"
    # Create calibration matrix
    ####
    cam_mat = np.array([[size[0] / (2 * np.tan(np.deg2rad(fov_x))), 0, size[0] / 2],
                        [0, size[1] / (2 * np.tan(np.deg2rad(fov_y))), size[1] / 2],
                        [0, 0, 1]])

    theta_x = np.rad2deg(0)
    theta_y = np.rad2deg(0)
    theta_z = np.rad2deg(0)
    x_rot = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    y_rot = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    z_rot = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])


    # unit 3d sphere
    pcl = create_sphere()

    # Create points cloud object
    if sphere:
        points_cloud_obj = points_cloud(pcd=pcl, eye_location=eye_location)
    else:
        points_cloud_obj = points_cloud(pcd=o3d.io.read_point_cloud(input_file), eye_location=eye_location)
    # Convert open3d format to numpy array
    point_cloud_in_numpy = np.asarray(points_cloud_obj.pcd.points)
    # img_points = np.empty_like(point_cloud_in_numpy)
    if points_cloud_obj.pcd.colors:
        point_color = np.asarray(points_cloud_obj.pcd.colors)
    else:
        point_color = points_cloud_obj.pcd.points
    # Points gradient
    for alpha in np.arange(0, end_angle, step_size):
        print(alpha)
        if manualProjection:
            image = points_cloud_obj.manualProjection(alpha)
        else:
            image = points_cloud_obj.openCVprojection(alpha)
        '''
        image2 = np.zeros((max(ys) - min(ys) + 1, max(xs) - min(xs) + 1, 3), np.uint8)
        if min(ys) < 0:
            ys = ys + (-min(ys))
        if min(xs) < 0:
            xs = xs + (-min(xs))
        for idx, i in enumerate(xs):
            image2[ys[idx],i] = (255, 255,255)
        '''
        # Gaussian Blur
        image = cv2.GaussianBlur(image, (3, 3), 0)

        print(f'writing image {alpha}')
        cv2.imshow('',image)
        cv2.waitKey()
        # cv2.imwrite("gradient/image" + str(alpha) + ".png", image)
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from open3d.open3d.geometry import TriangleMesh

class points_cloud:
    def __init__(self,pcd,eye_location):
        self.pcd = pcd
        self.eye_location = eye_location

    def visualize(self):
        # Visualize the point cloud within open3d
        mesh_frame = o3d.geometry.create_mesh_coordinate_frame(
            size=6, origin=eye_location)
        o3d.visualization.draw_geometries_with_custom_animation([self.pcd, mesh_frame])

    def openCVprojection(self,alpha):
        img_points = np.empty_like(self.pcd.points)
        rad = np.deg2rad(alpha)
        image = np.zeros((size[1], size[0], 3), np.uint8)
        rotation_mat = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
        img_points = cv2.projectPoints(point_cloud_in_numpy, rotation_mat, eye_location, cam_mat, distCoeffs=0,
                           aspectRatio=(size[0] / size[1]))
        img_points = np.asarray(img_points[0])
        xs = np.round([x[0][0] for x in img_points])
        ys = np.round([x[0][1] for x in img_points])
        xs = xs.astype(int)
        ys = ys.astype(int)

        for idx, i in enumerate(xs):
            if 0 <= i < len(image[:, ]) and 0 <= ys[idx] < len(image[0]):
                inv_norm_color = (point_color[idx] * 255).astype(int)
                image[i, ys[idx]] = inv_norm_color

        return image

    def manualProjection(self,alpha):
        rad = np.deg2rad(alpha)
        image = np.zeros((size[1], size[0], 3), np.uint8)
        zBuffer = np.zeros((size[1], size[0]), np.uint8)
        rotation_mat = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
        img_points = np.empty_like(self.pcd.points)
        T = np.column_stack([rotation_mat,eye_location])
        T = np.row_stack([T,[0,0,0,1]])
        C = np.matmul(cam_mat,hom_matrix)
        T = np.matmul(C,T)
        for idx,i in enumerate(point_cloud_in_numpy):
            x = np.append(i,1)
            x = np.matmul(T,x)
            img_points[idx][0] = x[0]/x[2]
            img_points[idx][1] = x[1]/x[2]
            img_points[idx][2] = x[2]

        tX = img_points[:,0]
        tY = img_points[:,1]
        tZ = img_points[:,2]
        xs = np.round(tX)
        ys = np.round(tY)
        xs = xs.astype(int)
        ys = ys.astype(int)

        for idx, i in enumerate(xs):
            if 0 <= i < len(image[:, ]) and 0 <= ys[idx] < len(image[0]):
                inv_norm_color = (point_color[idx] * 255).astype(int)
                if tZ[idx] > zBuffer[i, ys[idx]]:
                    image[i, ys[idx]] = inv_norm_color
                    zBuffer[i, ys[idx]] = tZ[idx]

        return image


if __name__ == '__main__':

    # NOTE: the code works on python 3.7 with open3d version 7.0.0

    # Initialization
    img_array = []
    size = (1280, 1024)
    fov_x = 60
    fov_y = 40
    end_angle = 360
    step_size = 0.5
    eye_location = np.array([10., -10., 10.])
    hom_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    # Read .ply file
    input_file = "spider.ply"
    # Create calibration matrix
    cam_mat = np.array([[size[1] /  (2 * np.tan(np.deg2rad(fov_y))), 0, size[1] / 2],
                        [0, size[0] / (2 * np.tan(np.deg2rad(fov_x))), size[0] / 2],
                        [0, 0, 1]])

    #Creating unit sphere (not finish)
    '''
    a = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=255)
    points_cloud_obj = points_cloud(pcd = a.vertices ,eye_location=eye_location)
    point_cloud_in_numpy = np.asarray(points_cloud_obj.pcd)
    o3d.visualization.draw_geometries([point_cloud_in_numpy])
    '''

    #Dont touch
    '''
    im_shape = (size[1],size[0],3)    
    chess = np.zeros(int((im_shape[0]*4),int(im_shape[1]*4),3),dtype=np.uint8)
    for i in range(int(chess.shape[0]/square_size)):
        for j in range(int(chess.shape[1]/square_size)):
            chess[square_size*j:square_size*(j+1),square_size*i:square_size*(i+1)] = np.random.randint(0,256,3,np.uint8)

    '''
    points_cloud_obj = points_cloud(pcd=o3d.io.read_point_cloud(input_file) ,eye_location=eye_location)
    # Convert open3d format to numpy array
    point_cloud_in_numpy = np.asarray(points_cloud_obj.pcd.points)
    if not points_cloud_obj.pcd.has_colors():
        # paints all points with a single a color because an error will occur if there's no color.
        points_cloud_obj.pcd.paint_uniform_color((0, 255, 0))
    point_color = np.asarray(points_cloud_obj.pcd.colors)
    img_points = np.empty_like(point_cloud_in_numpy)
    # Points projection
    for alpha in np.arange(0,end_angle,step_size):
        #image = points_cloud_obj.openCVprojection(alpha)
        image = points_cloud_obj.manualProjection(alpha)

        '''
        image2 = np.zeros((max(ys) - min(ys) + 1, max(xs) - min(xs) + 1, 3), np.uint8)
        if min(ys) < 0:
            ys = ys + (-min(ys))
        if min(xs) < 0:
            xs = xs + (-min(xs))
        for idx, i in enumerate(xs):
            image2[ys[idx],i] = (255, 255,255)
        '''

        image = cv2.GaussianBlur(image,(3,3),0)
        img_array.append(image)
        cv2.imwrite("projection/result" + str(alpha) + ".png", image)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("result11.mp4", fourcc, 30.0, size)
    for im in img_array:
        out.write(im)
    out.release()
    points_cloud_obj.visualize()
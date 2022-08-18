import cv2
import numpy as np

# field of view for x and y in degrees
fov_x = 45
fov_y = 30
# size of frame in the video
size = (360, 360)
# the angle in degrees in which the video stops (it always starts at 0)
end_angle = 45
# the change in angle in degrees in every frame
step_size = 1.5
# the size of the square on the chess board
square_size = 32
# the created video name
name = "created.mp4"

im_shape = (size[1], size[0], 3)
calib = np.array([[size[1] / (2 * np.tan(np.deg2rad(fov_y))), 0, size[1] / 2],
                  [0, size[0] / (2 * np.tan(np.deg2rad(fov_x))), size[0] / 2],
                  [0, 0, 1]])


def mapImage(im, T, sizeOutIm):
    """

    Args:
        im: image from which we create the mapping (type: np.array of size (n,m,3))
        T: matrix that defines the mapping (type: np.array of size (3,3))
        sizeOutIm: tuple that defines the size of the output image (type: tuple like (n',m',3)

    Returns:
        the mapped image
    """
    nim = np.zeros(sizeOutIm)  # the new image
    # create meshgrid of all coordinates in new image [x,y]
    x = np.arange(0, sizeOutIm[0])  # /sizeOutIm[0]
    y = np.arange(0, sizeOutIm[1])  # /sizeOutIm[1]
    ax, ay = np.meshgrid(x, y, indexing="xy")
    nim_coords = np.vstack((ax.ravel(), ay.ravel()))  # the coordinates of each pixel in the new image

    # add homogenous coord [x,y,1]
    nim_coords = np.vstack((nim_coords, np.ones((nim_coords.shape[1],))))
    source_coords = nim_coords.copy()
    # calculate source coordinates that correspond to [x,y,1] in new image
    source_coords = np.matmul(np.matmul(calib, np.matmul(np.linalg.inv(T), np.linalg.inv(calib))),
                              source_coords)  # apply the transform
    source_coords[:, :] /= source_coords[2, :]  # divide by the third coordinate (as required by homogenous coordinates)
    source_coords = np.delete(source_coords, 2, 0)  # delete the the homogenous coordinate
    nim_coords = np.delete(nim_coords, 2, 0)
    source_coords[0, :] = source_coords[0, :] + im.shape[0] / 2

    # find coordinates outside range and delete (in source and target)
    nim_coords = np.delete(nim_coords, np.where(source_coords[0, :] < 0), 1)
    source_coords = np.delete(source_coords, np.where(source_coords[0, :] < 0), 1)
    nim_coords = np.delete(nim_coords, np.where(source_coords[0, :] > im.shape[0] - 1), 1)
    source_coords = np.delete(source_coords, np.where(source_coords[0, :] > im.shape[0] - 1), 1)
    nim_coords = np.delete(nim_coords, np.where(source_coords[1, :] < 0), 1)
    source_coords = np.delete(source_coords, np.where(source_coords[1, :] < 0), 1)
    nim_coords = np.delete(nim_coords, np.where(source_coords[1, :] > im.shape[1] - 1), 1)
    source_coords = np.delete(source_coords, np.where(source_coords[1, :] > im.shape[1] - 1), 1)

    # interpolate - bilinear
    x_left = np.floor(source_coords[0, :]).astype(np.int)  # coordinates
    x_right = np.ceil(source_coords[0, :]).astype(np.int)
    y_top = np.floor(source_coords[1, :]).astype(np.int)
    y_bottom = np.ceil(source_coords[1, :]).astype(np.int)
    c_tl = im[x_left, y_top]  # colors
    c_tr = im[x_right, y_top]
    c_bl = im[x_left, y_bottom]
    c_br = im[x_right, y_bottom]
    x_delta = source_coords[0, :] - x_left
    y_delta = source_coords[1, :] - y_top
    c_top = x_delta[:, np.newaxis] * c_tr + (1 - x_delta)[:, np.newaxis] * c_tl
    c_bottom = x_delta[:, np.newaxis] * c_br + (1 - x_delta)[:, np.newaxis] * c_bl
    colors = y_delta[:, np.newaxis] * c_bottom + (1 - y_delta)[:, np.newaxis] * c_top  # the interpolated colors
    colors = np.round(colors)
    # apply corresponding coordinates
    nim_coords = nim_coords.astype(np.int)
    nim[nim_coords[0, :], nim_coords[1, :]] = colors
    return nim.astype(np.uint8)


def generate_pictures_2_angles(image, angle1, angle2, out_size):
    # Securing input for ben
    out_size = out_size.copy()
    # Checking Image Validity
    if type(image) == str:
        image = cv2.imread(image)
    else:
        assert (type(image) == np.ndarray and len(image.shape) == 3 and image.shape[2] == 3)

    image = cv2.resize(image, (int(out_size[0] * 10), int(out_size[1] * 10)))
    out_size.append(3)

    out1 = np.zeros(im_shape, dtype=np.uint8)
    rad = np.deg2rad(angle1)
    T = np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]])
    out1 = mapImage(image, T, im_shape)

    out2 = np.zeros(im_shape, dtype=np.uint8)
    rad = np.deg2rad(angle2)
    T = np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]])
    out2 = mapImage(image, T, out_size)

    return out1, out2


if __name__ == "__main__":
    from dictionary import MVMapping
    from block_matching import BlockMatching

    mv_dict = MVMapping()
    chess = np.zeros((im_shape[0] * 4, int(im_shape[1] * 4), 3), dtype=np.uint8)
    for i in range(int(chess.shape[0] / square_size)):
        for j in range(int(chess.shape[1] / square_size)):
            chess[square_size * j:square_size * (j + 1), square_size * i:square_size * (i + 1)] = np.random.randint(0,
                                                                                                                    256,
                                                                                                                    3,
                                                                                                                    np.uint8)
    for angle in range(0, 40):
        for step in range(0, 8):
            ref, cur = generate_pictures_2_angles(chess, angle, angle + step, [360,360])
            mv = BlockMatching.get_motion_vectors(cur, ref)
            mv_dict[mv] = step
    mv_dict.save_to("saved dictionaries/chess_plane_ego_rot_0-8_steps.pickle")
    # Test
    # out1, out2 = generate_pictures_2_angles("/home/txp2/RPI-BMA-RE/Dictionary/amogus.png", 10, 70, [450, 450])
    # cv2.imshow("out1", out1)
    # cv2.imshow("out2", out2)
    # cv2.waitKey()

    # img_array = []
    #
    # chess = np.zeros((im_shape[0] * 4, int(im_shape[1] * 4), 3), dtype=np.uint8)
    # for i in range(int(chess.shape[0] / square_size)):
    #     for j in range(int(chess.shape[1] / square_size)):
    #         chess[square_size * j:square_size * (j + 1), square_size * i:square_size * (i + 1)] = np.random.randint(0,
    #                                                                                                                 256,
    #                                                                                                                 3,
    #                                                                                                                 np.uint8)
    #
    # chess = cv2.imread("/home/txp2/RPI-BMA-RE/Dictionary/amogus.png")
    # chess = cv2.resize(chess, (im_shape[0] * 4, int(im_shape[1] * 4)))
    # # chess[int(1080-square_size/2):int(1080+square_size/2),1280-square_size:1280] = 255
    # # chess[720:720+square_size,1280-square_size:1280] = 255
    # # chess[1440-square_size:1440,1280-square_size:1280] = 255
    # for alpha in np.arange(0, end_angle, step_size):
    #     img = np.zeros(im_shape, dtype=np.uint8)
    #     rad = np.deg2rad(alpha)
    #     # T = np.array([[np.cos(rad),0,-np.sin(rad)],[0,1,0],[np.sin(rad),0,np.cos(rad)]])
    #     T = np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]])
    #     img = mapImage(chess, T, im_shape)
    #     img_array.append(img)
    #
    # print(f"Saving into {name}.mp4 ...")
    # out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, size)
    # frame = 0
    # for im in img_array:
    #     out.write(im)
    #     # cv2.imwrite("rot/Frame"+str(frame)+".jpeg", im)
    #     frame += 1
    # print("Saved")
    # out.release()

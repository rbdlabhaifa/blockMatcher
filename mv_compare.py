import math
import os
import sys
import glob

import numpy as np


# Utils
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def generate_vector_data_file(vec_array, file_path):
    f = open(file_path, "a")
    s = ''
    if len(vec_array) > 0:
        for mv in vec_array:
            s += f"{mv[0, 0]},{mv[0, 1]},{mv[1, 0]},{mv[1, 1]}|"
    f.write(s)
    f.close()


if __name__ == '__main__':

    url1 = "Benchmarks/Checkboard Video/mv_frame_data"
    url2 = "output"

    sum_of_mag_errors = 0
    sum_of_angle_errors = 0

    sum_of_mean_mag_errors = 0
    sum_of_mean_angle_errors = 0

    max_mag_error = 0
    max_angle_error = 0
    # print(frame_file_list1)
    frame = 0
    running = True
    while running:
        try:
            frame_file1 = open(url1 + "/frame" + str(frame) + ".txt", "r")
            frame_file2 = open(url2 + "/frame" + str(frame) + ".txt", "r")
        except FileNotFoundError as e:
            print(e)
            running = False
            continue
        frame += 1
        vectors1 = frame_file1.read().split("|")
        vectors2 = frame_file2.read().split("|")

        frame_file1.close()
        frame_file2.close()

        for j in range(len(vectors1) - 1):
            vec1 = np.array(vectors1[j].split(",")).astype(int)
            try:
                vec2 = np.array(vectors2[j].split(",")).astype(int)
            except ValueError as e:
                print(e)
                exit()
            mag_error = np.sqrt((int(vec2[2]) - int(vec2[0])) ** 2 + (int(vec2[3]) - int(vec2[1])) ** 2) - np.sqrt(
                (int(vec1[2]) - int(vec1[0])) ** 2 + (int(vec1[3]) - int(vec1[1])) ** 2)

            angle_error = round(angle_between(vec1, vec2))
            sum_of_mag_errors += mag_error
            sum_of_angle_errors += angle_error
            max_mag_error = max(max_mag_error, mag_error)
            max_angle_error = max(max_angle_error, angle_error)
        sum_of_mean_angle_errors += sum_of_angle_errors / len(vectors1)
        sum_of_mean_mag_errors += sum_of_mag_errors / len(vectors1)

    print(f"absolute_mean_mag_error = {sum_of_mean_mag_errors / (frame + 1)}")
    print(f"absolute_mean_angle_error = {sum_of_mean_angle_errors / (frame + 1)}")
    print(f"max_angle_error= {max_angle_error}")
    print(f"max_mag_error= {max_mag_error}")

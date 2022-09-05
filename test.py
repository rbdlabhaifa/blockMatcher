from typing import Tuple
import cv2
import numpy as np
from block_matching import BlockMatching


def calculate_focal_length(width: int, height: int, fov_x: float, fov_y: float) -> Tuple[float, float]:
    fx = (width / 2) / np.tan(fov_x / 2)
    fy = (height / 2) / np.tan(fov_y / 2)
    return fx, fy


def calculate_principle_point(width: int, height: int):
    # FIXME: This is probably wrong...    
    cx = width / 2
    cy = height / 2
    return cx, cy


def calculate_rotation(vector: Tuple[int, int, int, int],
                       fx: float, fy: float,
                       cx: float, cy: float) -> Tuple[float, float, float, float, float, float]:
    before_x, before_y, after_x, after_y = vector
    # FIXME: These inv_ variables are probably wrong...
    inv_fx, inv_fy = 1 / fx, 1 / fy
    inv_cx, inv_cy = 1 / cx, 1 / cy
    denominator = before_y * inv_fy + inv_cy
    numerator = np.sqrt(before_y * before_y * inv_fy * inv_fy + 2 * before_y * inv_cy * inv_fy + inv_cy * inv_cy + 1)
    x1 = 2 * np.arctan((numerator - 1) / denominator)
    x2 = -2 * np.arctan((numerator + 1) / denominator)
    denominator = -cx + fx * before_x * inv_fx + fx * inv_cx + after_x
    numerator1 = (cx - after_x) * (before_y * inv_fy + inv_cy)
    numerator2 = np.sqrt(
        cx * cx * before_y * before_y * inv_fy * inv_fy +
        2 * cx * cx * before_y * inv_cy * inv_fy +
        cx * cx * inv_cy * inv_cy +
        cx * cx -
        2 * cx * after_x * before_y * before_y * inv_fy * inv_fy -
        4 * cx * after_x * before_y * inv_cy * inv_fy -
        2 * cx * after_x * inv_cy * inv_cy -
        2 * cx * after_x -
        fx * fx * before_x * before_x * inv_fx * inv_fx -
        2 * fx * fx * before_x * inv_cx * inv_fx -
        fx * fx * inv_cx * inv_cx +
        after_x * after_x * before_y * before_y * inv_fy * inv_fy +
        2 * after_x * after_x * before_y * inv_cy * inv_fy +
        after_x * after_x * inv_cy * inv_cy +
        after_x * after_x
    )
    x3 = -2 * np.arctan((numerator1 - numerator2) / denominator)
    x4 = -2 * np.arctan((numerator1 + numerator2) / denominator)
    denominator = cy + fy * before_y * inv_fy + fy * inv_cy - after_y
    numerator1 = -cy * before_y * inv_fy - cy * inv_cy + fy + after_y * before_y * inv_fy + after_y * inv_cy
    numerator2 = np.sqrt(
        (cy * cy - 2 * cy * after_y + fy * fy + after_y * after_y) *
        (before_y * before_y * inv_fy * inv_fy + 2 * before_y * inv_cy * inv_fy + inv_cy * inv_cy + 1)
    )
    y1 = -2 * np.arctan((numerator1 + numerator2) / denominator)
    y2 = 2 * np.arctan((-numerator1 + numerator2) / denominator)
    return x1, x2, x3, x4, y1, y2


def main_check_arthur_formulas():
    video = '/home/rani/PycharmProjects/blockMatcher/Dictionary/data/360 video/1.mp4'
    motion_data = BlockMatching.extract_motion_data(
        extract_path='/home/rani/PycharmProjects/blockMatcher/Extra Code/extract motion data/motionVectors',
        video_path=video
    )
    width, height = 1200, 794
    fov_x, fov_y = 100, 100
    fx, fy = calculate_focal_length(width, height, fov_x, fov_y)
    cx, cy = calculate_principle_point(width, height)
    for i, frame in enumerate(motion_data):
        if i % 2 != 0:
            continue
        for vector in frame:
            print('real rotation =', 0.1 * (i / 2 + 1))
            print(calculate_rotation(vector, fx, fy, cx, cy))


if __name__ == '__main__':
    main_check_arthur_formulas()

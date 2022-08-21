from typing import List, Tuple
import numpy as np
from block_matching.cost_functions import sad
import cv2
from block_matching import BlockMatching
from dictionary import MVMapping


def cancel_noise(ref_frame: np.ndarray, cur_frame: np.ndarray,
                 mvs: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """
    Choose a vector that best represents the motion of the camera between two consecutive frames.

    :param ref_frame: The reference frame.
    :param cur_frame: The current frame.
    :param mvs: The motion vectors from the reference frame to the current frame.
    :return: The motion vector that represents the motion of the camera best.
    """
    # The best motion vector and its SAD to area ratio.
    best_mv_sad_to_area_ratio = float('inf')
    best_mv = None
    # A set of checked motion vectors.
    checked_mvs = set()
    for mv in mvs:
        # Get the (dx, dy) of the vector, it will be used to shift the reference frame.
        delta_x = mv[2] - mv[0]
        delta_y = mv[3] - mv[1]
        # Make sure (dx, dy) wasn't checked previously.
        if (delta_x, delta_y) in checked_mvs:
            continue
        checked_mvs.add((delta_x, delta_y))
        # Cut the rectangle that's in both the current frame and the shifted reference frame.
        if delta_y > 0:
            if delta_x > 0:
                shifted_ref_frame = ref_frame[:-delta_y, :-delta_x]
                shifted_cur_frame = cur_frame[delta_y:, delta_x:]
            elif delta_x == 0:
                shifted_ref_frame = ref_frame[:-delta_y]
                shifted_cur_frame = cur_frame[delta_y:]
            else:
                shifted_ref_frame = ref_frame[:-delta_y, -delta_x:]
                shifted_cur_frame = cur_frame[delta_y:, :delta_x]
        elif delta_y == 0:
            if delta_x > 0:
                shifted_ref_frame = ref_frame[:, :-delta_x]
                shifted_cur_frame = cur_frame[:, delta_x:]
            elif delta_x == 0:
                shifted_ref_frame = ref_frame
                shifted_cur_frame = cur_frame
            else:
                shifted_ref_frame = ref_frame[:, -delta_x:]
                shifted_cur_frame = cur_frame[:, :delta_x]
        else:
            if delta_x > 0:
                shifted_ref_frame = ref_frame[-delta_y:, :-delta_x]
                shifted_cur_frame = cur_frame[:delta_y, delta_x:]
            elif delta_x == 0:
                shifted_ref_frame = ref_frame[-delta_y:]
                shifted_cur_frame = cur_frame[:delta_y]
            else:
                shifted_ref_frame = ref_frame[-delta_y:, -delta_x:]
                shifted_cur_frame = cur_frame[:delta_y, :delta_x]
        # Calculate the SAD to area ratio of the rectangle.
        sad_ratio = sad(shifted_ref_frame, shifted_cur_frame) / (shifted_ref_frame.shape[0] *
                                                                 shifted_ref_frame.shape[1])
        if sad_ratio < best_mv_sad_to_area_ratio:
            best_mv_sad_to_area_ratio = sad_ratio
            best_mv = mv
    return best_mv


if __name__ == '__main__':
    dictionary = MVMapping('saved dictionaries/data1')
    for frame in range(6, 73):
        ref = cv2.imread(f'optitrack/data1/frame{frame - 1}.jpg')
        cur = cv2.imread(f'optitrack/data1/frame{frame}.jpg')
        mvs = BlockMatching.get_motion_vectors(cur, ref)
        best_mv = cancel_noise(ref, cur, mvs)
        print(f'best motion vector: {best_mv}')
        best_mv = (best_mv[2] - best_mv[0], best_mv[3] - best_mv[1])
        print(f'(dx, dy): {best_mv}')
        print(f'rotation: {dictionary[mvs]} degrees')
        mbs = BlockMatching.get_macro_blocks(ref)
        for i in range(len(mbs)):
            x, y = mbs[i][:2]
            w, h = mbs[i][2:]
            mbs[i] = (x + w // 2, y + h // 2, x + w // 2 + best_mv[0], y + h // 2 + best_mv[1])
        cur = ref.copy()
        for x1, y1, x2, y2 in mvs:
            ref = cv2.arrowedLine(ref, (x1, y1), (x2, y2), (0, 255, 0), 1)
        for x1, y1, x2, y2 in mbs:
            cur = cv2.arrowedLine(cur, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imshow('normal', ref)
        cv2.imshow('clean', cur)
        cv2.waitKey()
        cv2.destroyAllWindows()

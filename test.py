import os.path

if __name__ == '__main__':

    from python_block_matching.algorithms import *
    from python_block_matching.cost_functions import *
    from python_block_matching.utils import *

    import cv2

    video_path = '../rawVideo.mp4'
    assert os.path.exists(video_path)
    video = cv2.VideoCapture(video_path)
    was_read, frame1 = video.read()
    cv2.imshow('Press SPACE to continue.', frame1)
    cv2.waitKey(20000)
    while was_read:
        was_read, frame2 = video.read()
        assert was_read
        for start in get_all_center_points(frame2, 32):
            target = three_step_search(frame2, frame1, start[0], start[1], 32, 'MSE')
            frame2 = cv2.arrowedLine(frame2, start, target, (0, 255, 0), thickness=4)
        cv2.imshow('Press SPACE to continue.', frame2)
        cv2.waitKey(20000)
        was_read, frame1 = video.read()
        cv2.imshow('Press SPACE to continue.', frame1)
        cv2.waitKey(20000)

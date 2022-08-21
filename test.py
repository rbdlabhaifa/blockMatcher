from block_matching import *
import cv2


if __name__ == '__main__':
    mvs = BlockMatching.extract_motion_vectors('/home/rani/Desktop/amir.txt')
    mbs = BlockMatching.extract_macro_blocks('/home/rani/Desktop/amir.txt')
    video = BMVideo('/home/rani/Desktop/output.mp4')
    for f in range(1, video.get_frame_count() - 1):
        frame1 = video[f - 1].base_image
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        frame2 = video[f].base_image
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        mv = mvs[f - 1]
        fra = BMFrame(frame2 - frame1)
        fra.draw_motion_vector(mv, (255, 0, 0), 1)
        fra.show()
        # mb = mbs[f - 1]
        # cf = BlockMatching.form_compensated_frame(frame1.base_image, mb, mv)
        # cv2.imshow('2', frame2.base_image)
        # cv2.imshow('com', cf)

        cv2.waitKey()


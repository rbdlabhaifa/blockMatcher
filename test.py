import os
import cv2
from block_matching import BlockMatching, BMFrame


def draw_vectors():
    for video in os.listdir('/home/rani/Downloads'):
        if not video.endswith('mp4'):
            continue
        if not os.path.exists(f'/home/rani/Downloads/{video[:-4]}_folder'):
            os.mkdir(f'/home/rani/Downloads/{video[:-4]}_folder')
        mvs = BlockMatching.extract_motion_vectors(f'/home/rani/Downloads/{video[:-4]}.txt')
        cap = cv2.VideoCapture(f'/home/rani/Downloads/{video}')
        prev = 0
        print(f'doing {video} ...')
        for framenum, mv in mvs.items():
            if framenum == prev:
                was_read, frame = cap.read()
                if not was_read:
                    break
                frame = BMFrame(frame)
                frame.draw_motion_vectors(mv, (0, 0, 255), 1)
                cv2.imwrite(f'/home/rani/Downloads/{video[:-4]}_folder/{framenum}.png', frame.drawn_image)
                prev += 1
            else:
                prev = framenum
                was_read, frame = cap.read()
                if not was_read:
                    break
                cv2.imwrite(f'/home/rani/Downloads/{video[:-4]}_folder/{framenum}.png', frame)


if __name__ == '__main__':
    draw_vectors()
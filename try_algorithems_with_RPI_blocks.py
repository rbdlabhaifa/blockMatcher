from python_block_matching import algorithms, cost_functions, utils
import cv2

vid_url = "almost360.mp4"
csv_url = "almost360.csv"
frame_offset = 1

cap = cv2.VideoCapture(vid_url)

for i in range(frame_offset):
    ret, frame = cap.read()
last_frame = frame
frame_our = frame.copy()
frame_rasp = frame.copy()
with open(csv_url, "r") as file:
    file.readline()  # Header Line of CSV
    last_frame_num = 0
    for macro_block in file:
        framenum, source, blockw, blockh, srcx, srcy, dstx, dsty, flags = macro_block.replace(" ", "").split(",")
        if not int(framenum) == last_frame_num:
            cv2.imshow("ours", frame_our)
            cv2.imshow("rasp", frame_rasp)
            cv2.waitKey()
            last_frame_num = int(framenum)
            last_frame = (frame.copy())
            ret, frame = cap.read()
            frame_our = frame.copy()
            frame_rasp = frame.copy()
            frame_calc = frame.copy()
            if not ret:
                break
        coords = algorithms.diamond_search(frame_calc, last_frame, int(dstx) - int(blockw) // 2,
                                           int(dsty) - int(blockh) // 2,
                                           (int(blockw), int(blockh)), cost_function="SAD")

        cv2.arrowedLine(frame_our,
                                (int(dstx), int(dsty)), (int(coords[0]), int(coords[1])), (0, 255, 0), 1)

        cv2.arrowedLine(frame_rasp,
                                (int(dstx), int(dsty)),
                                (int(srcx), int(srcy)), (0, 0, 255), 1)

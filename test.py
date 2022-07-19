from python_block_matching import *

vid = BMVideo('Lab Testing Phase 1/output.avi')
vecs = BlockMatching.extract_motion_vectors('Lab Testing Phase 1/out.txt')
for i in range(40, vid.get_frame_count() + 500):
    frame1, frame2 = vid[i - 1], vid[i]
    if i in vecs.keys():
        mvs = vecs[i]
        frame1.draw_motion_vector(mvs, color=(255, 0, 0), thickness=2)
    frame1.show()
frame2.show()

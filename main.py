import cv2

ref_image = ""
cur_image = ""

image = cv2.imread(ref_image)
image = cv2.arrowedLine(image, (image.shape[0], image.shape[1]), three_step_search(), (255, 0, 0), 3)
cv2.imshow(image)

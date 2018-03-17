import cv2
import os

IMAGE_PATH = "/Users/ksk/img/metrics_newdata/new_slides/"




videoFrameNames = [ filename for filename in os.listdir(IMAGE_PATH) if filename.endswith(".png") ]


for vfn in videoFrameNames:
    img = cv2.imread(IMAGE_PATH + vfn, 0)
    small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    cv2.imwrite(IMAGE_PATH + vfn, small)

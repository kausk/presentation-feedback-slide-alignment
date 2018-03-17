import cv2
import numpy as np
import os
from os import listdir



## imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
image = cv2.imread("../shortened_video/50000-vid.jpg")
dimensions = cv2.selectROI(image)


## load files in shortened_video
files = os.listdir("../shortened_video")
for f in files:
    img = cv2.imread("../shortened_video/" + f, 0)
    cropped = img[int(dimensions[1]):int(dimensions[1]+dimensions[3]), int(dimensions[0]):int(dimensions[0]+dimensions[2])]
    cv2.imwrite("../slides_only_video/" + f, cropped)



## crop according to dimensions



## save in slides_only_video



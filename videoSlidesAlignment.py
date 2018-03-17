## for each slide
## https://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html
## https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
## https://docs.opencv.org/3.1.0/d4/dc6/tutorial_py_template_matching.html


## track 10 nearest slides

import numpy as np
import cv2
import imutils

videoImage = cv2.imread("video.png", 0)
edgesVideoImage = cv2.Canny(videoImage,100,200)

template = cv2.imread("template.png", 0)
templateEdges = cv2.Canny(template,100,200)

w, h = template.shape[::-1]


match = cv2.matchTemplate(edgesVideoImage, templateEdges, cv2.TM_CCOEFF_NORMED)

## from https://docs.opencv.org/3.1.0/d4/dc6/tutorial_py_template_matching.html
threshold = 0.2
loc = np.where( abs(match) >= threshold)
for pt in zip(*loc[::-1]):
     cv2.rectangle(edgesVideoImage, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
print(loc)
cv2.imshow('image', templateEdges)
cv2.waitKey(0)
cv2.destroyAllWindows()
## if not confident search through all slides and find most similar

## 
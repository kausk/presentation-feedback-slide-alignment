## Feature matching method from: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher

import cv2
from matplotlib import pyplot as plt
import numpy as np

## IMG
videoImage = cv2.imread("video.png", 0)
template = cv2.imread("template.png", 0)

## Keypoints
featORB = cv2.ORB_create()
videoKP, vidDesc = featORB.detectAndCompute(videoImage, None)
templateKP, tempDesc = featORB.detectAndCompute(videoImage, None)

bruteF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

## matching
matches = bruteF.match(vidDesc,tempDesc)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
matchedImgs = cv2.drawMatches(videoImage,videoKP,template,templateKP,matches[:10], None, flags=2)

## Display  
plt.imshow(matchedImgs)
plt.show()

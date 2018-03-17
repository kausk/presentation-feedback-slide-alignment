import cv2
import os
import pickle
from utilities import visualize, drawMatches, predictARGMIN
from utilities import calculateKeypoints, serializableKeypts, slidesInRange, convertPickledToKPDesc, computeRDistanceDifference, computeSumDiffKeypoints
import numpy as np
from matplotlib import pyplot as plt

slideImgPath = "./360-video-10-23/"


slideNames = [ filename for filename in os.listdir( slideImgPath) if filename.endswith(".jpg")] 
slideNames = slidesInRange(slideNames, 0, 74)

img = cv2.imread(slideImgPath + slideNames[0])

for i in range(1, 74):
    img2 = cv2.imread(slideImgPath + slideNames[i])
    img = np.concatenate((img, img2), axis=1)

cv2.imwrite('images_horizontal.png', img)

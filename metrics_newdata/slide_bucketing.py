import cv2
import os
import pickle
from utilities import visualize, drawMatches, predictARGMIN
from utilities import calculateKeypoints, serializableKeypts, slidesInRange, convertPickledToKPDesc, computeRDistanceDifference, computeSumDiffKeypoints
import numpy as np
from matplotlib import pyplot as plt

## plot number of matches
## where to load/save
slideImgPath = "./360-video-10-23/"

## Load slides, compute keypoint matching to following slide
slideNames = [ filename for filename in os.listdir( slideImgPath) if filename.endswith(".jpg")] 
slideNames = slidesInRange(slideNames, 0, 74)

## load dictionary of things
slideImgKeyptsFile = "./slide_keypts/slideKeyptsDesc.pkl"
slideImgKeypts = pickle.load( open( slideImgKeyptsFile, "rb" ) )

print(slideImgKeypts.keys())

slideImgKeyptsDesc = dict()

## load slide img, keypoints
for sn in slideNames:
    path = slideImgPath + sn
    slideImgKeyptsDesc[sn] = convertPickledToKPDesc(slideImgKeypts[path])



FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

## matches is a list of DMatch objects
flann = cv2.FlannBasedMatcher(index_params, search_params)

matchSlideFrameDict = dict()

matchMetric = dict()
matchMetricNotNormalized = dict()

numMatches = []

for prev_slide in range(0, 73):
    next_slide = prev_slide + 1
    print("matching")
    slidekpts, slideDescs = slideImgKeyptsDesc[slideNames[prev_slide]]
    next_slidekpts, next_slideDescs = slideImgKeyptsDesc[slideNames[next_slide]]
    matches = flann.knnMatch(np.asarray(slideDescs,np.float32),np.asarray(next_slideDescs,np.float32), k=2)
    matchPairs = []
    distances = []
    good = []
    slideKeypoints = []
    frameKeypoints = []

    good = []


    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    
    print("prev_slide: ", prev_slide)
    print("next_slide", next_slide)
    print("good: ", len(good))
    numMatches.append(len(good))

plt.scatter(np.arange(0, 73, 1.0), np.array(numMatches))
plt.xticks(np.arange(0, 75, 1.0))
plt.show()
## visualize

## if above a certain number of misses, 
## prev to current is one boundary
## set as one slide
## move to next slide

## visualize all contiguous slides
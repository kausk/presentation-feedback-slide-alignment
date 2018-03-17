import cv2
import os
import pickle
from utilities import visualize, predictARGMIN
from utilities import calculateKeypoints, serializableKeypts, slidesInRange, convertPickledToKPDesc
import numpy as np

## where to load/save
slideImgPath = "./new_slides/"
vidFramePath = "./newdata_vid_frames_sampled_fullframe/png/"

slideImgKeyptsPath = "./new_slide_keypts/"
vidImgKeyptsPath = "./new_vid_keypts/"

slideImgKeyptsFile = "./new_slide_keypts/slideKeyptsDesc.pkl"
vidImgKeyptsFle = "./new_vid_keypts/vidKeyptsDesc.pkl"


matchesPath = "./desc_matches/"


VID_START = 0
VID_END = 22
SLIDE_START = 0
SLIDE_END = 12
NUM_SLIDES = VID_END - VID_START - 1

## getting filenames
videoFrameNames = [ filename for filename in os.listdir(vidFramePath) if filename.endswith(".png") ]
videoFrameNames = slidesInRange(videoFrameNames, VID_START, VID_END)

slideNames = [ filename for filename in os.listdir( slideImgPath) if filename.endswith(".png")] 
slideNames = slidesInRange(slideNames, SLIDE_START, SLIDE_END)

sampleImgSlides = cv2.imread(slideImgPath + slideNames[0], 0)
sampleVidSlides = cv2.imread(vidFramePath + videoFrameNames[0], 0)


## load dictionary of things
slideImgKeypts = pickle.load( open( slideImgKeyptsFile, "rb" ) )
vidImgKeypts = pickle.load( open( vidImgKeyptsFle,  "rb" ) )

slideImgKeyptsDesc = dict()
vidImgKeyptsDesc = dict()

## load slide img, keypoints
for sn in slideNames:
    path = slideImgPath + sn
    slideImgKeyptsDesc[sn] = convertPickledToKPDesc(slideImgKeypts[path])


for vidName in videoFrameNames:
    path = vidFramePath + vidName
    vidImgKeyptsDesc[vidName] = convertPickledToKPDesc(vidImgKeypts[path])

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

## matches is a list of DMatch objects
flann = cv2.FlannBasedMatcher(index_params, search_params)

matchSlideFrameDict = dict()

matchMetric = dict()
matchMetricNotNormalized = dict()

print(slideNames)
print(videoFrameNames)

## cross matching
matchID = 0
for slide in range(0, NUM_SLIDES):
    for frame in range(0, NUM_SLIDES):

        slidekpts, slideDescs = slideImgKeyptsDesc[slideNames[slide]]
        framekpts, frameDescs = vidImgKeyptsDesc[videoFrameNames[frame]]
        matches = flann.knnMatch(np.asarray(slideDescs,np.float32),np.asarray(frameDescs,np.float32), k=2)

        matchPairs = []
        distances = []
        good = []
        slideKeypoints = []
        frameKeypoints = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:

                slideKeypoints.append(slidekpts[m.queryIdx])
                frameKeypoints.append(framekpts[m.trainIdx])
                distances.append(m.distance)
        matchID += 1
        distances.sort(reverse=True)

        matchMetric[(slide, frame)] = sum(distances[0:10]) / len(matches)
        matchMetricNotNormalized[(slide, frame)] = sum(distances[0:10]) 
        matchSlideFrameDict[(slide, frame)  ] = matchPairs


visualize(matchMetric, NUM_SLIDES, "./new_results/normalized_distanceReversed.jpg")
visualize(matchMetricNotNormalized, NUM_SLIDES, "./new_results/distanceReversed.jpg")

predictARGMIN(matchMetric)
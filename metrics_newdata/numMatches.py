import cv2
import os
import pickle
from utilities import visualize
from utilities import calculateKeypoints, serializableKeypts, slidesInRange, convertPickledToKPDesc
import numpy as np

## where to load/save
slideImgPath = "./360-video-10-23/"
vidFramePath = "./vid_frames_sampled/"

slideImgKeyptsFile = "./slide_keypts/slideKeyptsDesc.pkl"
vidImgKeyptsFle = "./vid_keypts/vidKeyptsDesc.pkl"

matchesPath = "./desc_matches/"


VID_START = 0
VID_END = 23
SLIDE_START = 51
SLIDE_END = 74
NUM_SLIDES = VID_END - VID_START - 1

## getting filenames
videoFrameNames = [ filename for filename in os.listdir(vidFramePath) if filename.endswith(".jpg") ]
videoFrameNames = slidesInRange(videoFrameNames, VID_START, VID_END)

slideNames = [ filename for filename in os.listdir( slideImgPath) if filename.endswith(".jpg")] 
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
                good.append(m)
                distances.append(m.distance)
        matchID += 1
        distances.sort(reverse=True)

        print(distances)
        matchMetric[(slide, frame)] = len(good)
        matchSlideFrameDict[(slide, frame)  ] = matchPairs

visualize(matchMetric, NUM_SLIDES, "./numFriends.jpg")

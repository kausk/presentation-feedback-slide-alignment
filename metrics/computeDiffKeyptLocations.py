import cv2
import os
import pickle
from utilities import visualize, drawMatches
from utilities import calculateKeypoints, serializableKeypts, slidesInRange, convertPickledToKPDesc, computeRDistanceDifference, computeSumDiffKeypoints
import numpy as np

## where to load/save
slideImgPath = "../slides/"
vidFramePath = "../bb_sampled_video_frames/"

slideImgKeyptsFile = "./slide_keypts/slideKeyptsDesc.pkl"
vidImgKeyptsFle = "./vid_keypts/vidKeyptsDesc.pkl"

matchesPath = "./desc_matches/"


## getting filenames
videoFrameNames = [ filename for filename in os.listdir(vidFramePath) if filename.endswith(".jpg") ]
videoFrameNames = slidesInRange(videoFrameNames, 0, 28)

slideNames = [ filename for filename in os.listdir( slideImgPath) if filename.endswith(".png")] 
slideNames = slidesInRange(slideNames, 2, 30)

sampleImgSlides = cv2.imread(slideImgPath + slideNames[0], 0)
sampleVidSlides = cv2.imread(vidFramePath + videoFrameNames[0], 0)


## slide and video dimensions
slideDimensions = (sampleImgSlides.shape[0], sampleImgSlides.shape[1])
vidDimensions = (sampleVidSlides.shape[0], sampleVidSlides.shape[1])

## load dictionary of things
slideImgKeypts = pickle.load( open( slideImgKeyptsFile, "rb" ) )
vidImgKeypts = pickle.load( open( vidImgKeyptsFle,  "rb" ) )
print(slideImgKeypts.keys())

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

matchMetricNotNormalized = dict()
matchMetric = dict()


## cross matching
matchID = 0
for slide in range(0, 28):
    for frame in range(0, 28):

        slidekpts, slideDescs = slideImgKeyptsDesc[slideNames[slide]]
        framekpts, frameDescs = vidImgKeyptsDesc[videoFrameNames[frame]]
        matches = flann.knnMatch(np.asarray(slideDescs,np.float32),np.asarray(frameDescs,np.float32), k=2)

        matchPairs = []
        distances = []
        good = []
        slideKeypoints = []
        frameKeypoints = []

        good = []


        for m, n in matches:
            if m.distance < 0.75*n.distance:
                slideKeypoints.append(slidekpts[m.queryIdx].pt)
                frameKeypoints.append(framekpts[m.trainIdx].pt)
                good.append(m)
            
        '''
        if (frame == slide):
            slideImg = cv2.imread(slideImgPath + slideNames[slide], 0)
            vidImg = cv2.imread(vidFramePath + videoFrameNames[frame], 0)
            
            drawMatches(slidekpts, framekpts, slideImg, vidImg, good)
        '''


        matchMetric[(slide, frame)] = computeSumDiffKeypoints(slideKeypoints, frameKeypoints, slideDimensions, vidDimensions)
        matchMetricNotNormalized[(slide, frame)] = computeSumDiffKeypoints(slideKeypoints, frameKeypoints, slideDimensions, vidDimensions, False)

visualize(matchMetric, 28, "./sumDistanceKeypoints.jpg")
visualize(matchMetric, 28, "./sumDistanceKeypoints.jpg")
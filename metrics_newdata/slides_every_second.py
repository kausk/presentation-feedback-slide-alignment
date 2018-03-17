import cv2
import os
import pickle
from utilities import visualize, drawMatches, predictARGMIN, saveMatches
from utilities import calculateKeypoints, serializableKeypts, slidesInRange, convertPickledToKPDesc, computeSumDiffKeypoints
import numpy as np
from matplotlib import pyplot as plt

SLIDE_PATH = "./360-video-10-23/"
TARGET_SLIDES = [54, 58, 61, 67, 71, 72]
VIDEO_PATH = "./subsampled_slides_video/cropped_shortened_pres.mp4"
VIDEO_START_SEC = 0
VIDEO_END_SEC = 63
VIDEO_FRAME_PATH = "./vid_frames_sampled/"
SAVE_MATCH_PATH = "./frames_slides_matches/"


slideNames = [ filename for filename in os.listdir( SLIDE_PATH) if filename.endswith(".jpg")] 
slideNames = slidesInRange(slideNames, 0, 74)
videoFrameNames = [ filename for filename in os.listdir(VIDEO_FRAME_PATH) if filename.endswith(".jpg") ]
videoFrameNames = slidesInRange(videoFrameNames, 0, 74)

## slide and video dimensions
sampleImgSlides = cv2.imread(SLIDE_PATH + slideNames[0], 0)
sampleVidSlides = cv2.imread(VIDEO_FRAME_PATH + videoFrameNames[0], 0)
slideDimensions = (sampleImgSlides.shape[0], sampleImgSlides.shape[1])
vidDimensions = (sampleVidSlides.shape[0], sampleVidSlides.shape[1])


slideImgKeyptsFile = "./slide_keypts/slideKeyptsDesc.pkl"
vidImgKeyptsFle = "./vid_keypts/vidKeyptsDesc.pkl"
slideImgKeypts = pickle.load( open( slideImgKeyptsFile, "rb" ) )


slideImgKeyptsDesc = dict()
vidImgKeyptsDesc = dict()

print(slideImgKeypts.keys())
## load slide img, keypoints
for sn in slideNames:
    path = SLIDE_PATH + sn
    slideImgKeyptsDesc[sn] = convertPickledToKPDesc(slideImgKeypts[path])


seconds = []
time = VIDEO_START_SEC*1000

while time < VIDEO_END_SEC*1000:
    seconds.append(time)
    time += 1000

matrixSlidesFrames = np.zeros(shape=(len(TARGET_SLIDES), len(seconds)))


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

## matches is a list of DMatch objects
flann = cv2.FlannBasedMatcher(index_params, search_params)

video = cv2.VideoCapture(VIDEO_PATH)

for idx1, slide in enumerate(TARGET_SLIDES):
    print(slideNames[slide])
    slidekpts, slideDescs = slideImgKeyptsDesc[slideNames[slide]]
    slideImg = cv2.imread(SLIDE_PATH + slideNames[slide])


    for idx2, sec in enumerate(seconds):
        video.set(cv2.CAP_PROP_POS_MSEC, sec)
        success, videoImg = video.read()
        ### compute matches between img and videoImg
        framekpts, frameDescs = calculateKeypoints(videoImg)

        ## evaluate using matrix and normalized metric
        matches = flann.knnMatch(np.asarray(slideDescs,np.float32),np.asarray(frameDescs,np.float32), k=2)
        
        
        slideKeypoints = []
        frameKeypoints = []
        goodMatches = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                slideKeypoints.append(slidekpts[m.queryIdx].pt)
                frameKeypoints.append(framekpts[m.trainIdx].pt)
                goodMatches.append(m)

        ## save match in folder
        matrixSlidesFrames[idx1][idx2] = computeSumDiffKeypoints(slideKeypoints, frameKeypoints, slideDimensions, vidDimensions)
        saveMatches(slidekpts, framekpts, slideImg, videoImg, goodMatches, SAVE_MATCH_PATH + str(sec) + '-' + str(slide) + '-' + 'match.jpg')


prevGuess = 0
for idx1, slide in enumerate(TARGET_SLIDES):
    possibleFrames = list(matrixSlidesFrames[idx1, :])
    possibleFramesTrimmed = possibleFrames[prevGuess:]
    print(prevGuess)
    print(len(possibleFrames))
    framePrediction = min(possibleFramesTrimmed)
    origFrameNumber = possibleFrames.index(framePrediction)
    prevGuess = origFrameNumber
    print("Slide " + str(slide) + " is on frame " + str(origFrameNumber))

## Display match matrix
fig = plt.figure()
plt.imshow(matrixSlidesFrames)
plt.colorbar()
plt.xticks(np.arange(0, len(seconds), 1.0), fontsize=5)
plt.yticks(np.arange(0, len(TARGET_SLIDES), 1.0), fontsize=5)
plt.show()
fig.savefig("./slides_frames_diffkpts_match.png")
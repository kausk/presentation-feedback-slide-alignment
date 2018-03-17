import cv2
import os
import pickle
from utilities import calculateKeypoints, serializableKeypts, slidesInRange, convertPickledToKPDesc

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

## cross matching
for slide in range(2, 30):
    for frame in range(0, 28):

        slideDescs = slideImgKeyptsDesc[slideNames[slide-2]][1]
        frameDescs = vidImgKeyptsDesc[videoFrameNames[frame]][1]

        matches = flann.knnMatch(slideDescs,frameDescs,k=2)


        matchPairs = []
        for pair in matches:
            pairData = []
            for p in pair:
                matchData = [p.queryIdx, p.trainIdx, p.imgIdx, p.distance]
                pairData.append(matchData)
            matchPairs.append(pairData)
        
        matchSlideFrameDict[(slide-2, frame)] = matchPairs
        
pickleOutput = open(matchesPath + "matches.pkl", 'wb')
pickle.dump(matchSlideFrameDict, pickleOutput)


## load vid img, keypoints

## do a 28x28 cross matching of them

## save each (img1, img2, crossMatching)
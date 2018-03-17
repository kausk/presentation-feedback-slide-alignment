import cv2
from os import listdir
from cvSIFTFN_save import saveMatchingImgs, frameNumImgs

import numpy as np

import matplotlib.pyplot as plt


## 1. load slide images
SLIDE_FOLDER = "../slides"

VIDEO_PATH = "../data/mackenzie-talk-1.MP4"

BB_VIDEO_PATH = "../data/croppedBBslides4.mp4"

VIDEO_SAMPLED_FOLDER = "../sampled_video_frames"

BB_VIDEO_SAMPLED_FOLDER = "../bb_sampled_video_frames"

WRITE_MATCH_DIRECTORY = "../matched_slides_bb_video"

## slides   
slideNames = [ filename for filename in listdir(SLIDE_FOLDER) if filename.endswith(".png")] 

slideImages = []
slideImagesDict = dict()
for i in range(2, 30): ## only first 30 slides in the video
    slideImg = cv2.imread(SLIDE_FOLDER + "/" + slideNames[i], 0)
    slideImages.append(slideImg)
    slideImagesDict[i-2] = slideImg
    

print("slide images", len(slideImages))
## 2. subsample videos at specified intervals
## timings from https://docs.google.com/document/d/1kjdn-BpeQjfIYArPKgCjmUUzLT2vI0Btk7CDOqBFptw/edit
sampleAtSeconds = [14000, 31000, 60000, 90000, 120000, 130000, 180000, 210000, 237000, 242000, 260000, 300000 ,310000, 330000, 360000, 390000, 420000, 450000, 510000, 550000, 570000, 600000, 630000, 660000, 700000, 720000, 741000, 750000]
video = cv2.VideoCapture(BB_VIDEO_PATH)

videoFrames = []
videoFramesDict = dict()
index = 0
for secSnapshot in sampleAtSeconds:
    video.set(cv2.CAP_PROP_POS_MSEC, secSnapshot)
    success, img = video.read()
    if success:
        path = BB_VIDEO_SAMPLED_FOLDER + "/" + str(index) + "-" +str(secSnapshot) + ".jpg"

        x = cv2.imwrite(path, img)
        videoFrames.append(img)
        videoFramesDict[index] = img
        index += 1
    else:
        print("not a success")
print("slide images", len(videoFrames))

## 3. create map between slide img and subsampled videos
## match each slide, subsampled, save using CVSiftFunc
for i in range(0, 28):
    path = WRITE_MATCH_DIRECTORY + "/matching-" + str(i)
    try:
        saveMatchingImgs(slideImages[i], videoFrames[i], path )
    except:
        print("")
        ## do nothing

## 4. plug in bounding box crop part


## 5. match img to slide -- print out 3 metrics

## 6. create confusion matrix
matchMatrix = np.zeros(shape=(28, 28))
imgMatrixDict = dict()
for i in range(28):
    imgMatrixDict[i] = dict()
    for j in range(28):
        print(i, j)
        try:
            print("success")
            numMatchesIJ, imgMatchIJ = frameNumImgs(slideImagesDict[i], videoFramesDict[j])
            print("num matches", numMatchesIJ)
            matchMatrix[i][j] = numMatchesIJ
            imgMatrixDict[i][j] = imgMatchIJ
        except:
            print("not enough matches")
            matchMatrix[i][j] = 0
            imgMatrixDict[i][j] = None


print("matches complete")
fig = plt.figure()
plt.imshow(matchMatrix)
plt.colorbar()
plt.xticks(np.arange(0, 28, 1.0))
plt.yticks(np.arange(0, 28, 1.0))
plt.show()
fig.savefig("numMatchesMatrix.png")



from __future__ import division
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


## from https://stackoverflow.com/questions/5407969/distance-formula-between-two-points-in-a-list
def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

## returns img with keypoints, descriptors
def calculateKeypoints(img):
    surfDetector = cv2.xfeatures2d.SIFT_create()
    videoImageKP, desV = surfDetector.detectAndCompute(img, None)
    return videoImageKP, desV

## help from http://hanzratech.in/2015/01/16/saving-and-loading-keypoints-in-file-using-opencv-and-python.html
def serializableKeypts(keys, descs):
    if (len(keys) == 0 or len(descs) == 0):
        return []
    keysDescs = []
    
    for key, desc in zip(keys, descs):
        keysDescs.append( (key.pt, key.size, key.angle, key.response, key.octave, key.class_id, desc) )
    
    return keysDescs

def slidesInRange(slidenames, start, end):
    slideNameOrderDict = dict()
    ## storing '23-slide.png' to just 23
    for sn in slidenames:
        snIndex = sn.split('-')
        slideNameOrderDict[int(snIndex[0])] = sn

    return [slideNameOrderDict[i] for i in sorted(slideNameOrderDict.keys())][start:end]

def convertPickledToKPDesc(dataTuple):
    keyPts = []
    descS  = []
    for dt in dataTuple:
        keyPoint = cv2.KeyPoint( x = dt[0][0],
                                y = dt[0][1],
                                _size=dt[1],
                                _angle=dt[2],
                                _response = dt[3],
                                _octave = dt[4],
                                _class_id = dt[5]
                            )
        desc = dt[6]
        keyPts.append(keyPoint)
        descS.append(desc)

    return keyPts, descS


## convert list DM Pairs
def convertListDMPairs(DMPairs):
    DMPairs = []
    for pair in DMPairs:
        pairDM = []
        for p in pair:
            DM = cv2.DMatch(p[0], p[1], p[2], p[3])
            pairDM.append(DM)
        DMPairs.append(pairDM)
    return DMPairs


## load x, y dict of DMatch
def convertXYMatch(filename):
    listPairs = pickle.load( open( filename, "rb" ) )
    DMMatchDict = dict()
    for k in listPairs.keys():
        DMMatchDict[k] = convertListDMPairs(listPairs[k])
    return DMMatchDict


## return (x, y)
## evaluate matchings
def evaluateMetricXY(matchesDict, metricFN, numSamples):
    metricsSlideXFrameY = dict()
    for slide in range(numSamples):
        for videoFrame in range(numSamples):
            DMMatchPairs = DMMatchDict[(slide, videoFrame)]
            metricsSlideXFrameY[slide][videoFrame] = metricFN(DMMatchPairs)
    return metricsSlideXFrameY


## visualize the effectiveness of this metric
def visualize(metricsXY, numRows, savePath):
    matchMatrix = np.zeros(shape=(numRows, numRows))
    for slide in range(numRows):
        for frame in range(numRows):
            matchMatrix[slide][frame] = metricsXY[(slide, frame)]
            
    fig = plt.figure()
    plt.imshow(matchMatrix)
    plt.colorbar()
    plt.xticks(np.arange(0, numRows, 1.0))
    plt.yticks(np.arange(0, numRows, 1.0))
    plt.show()
    fig.savefig(savePath)



def getRelativeXY(point, dimensions):
    return ( (point[0] / dimensions[0]), (point[1]/dimensions[1]) )

def computeSumDiffKeypoints(pointsSlide, pointsFrame, dimSlides, dimFrame):
    if (len(pointsSlide) == 0 or len(pointsSlide) == 0):
        return 1
    difference = 0
    for slide, frame in zip(pointsSlide, pointsFrame):
        difference += distance(getRelativeXY(slide, dimSlides), getRelativeXY(frame, dimFrame))
    toReturn = -1 * difference 
    if toReturn > -50:
        return difference
    else:
        return -50

def computeRDistanceDifference(pointsSlide, pointsFrame, dimSlides, dimFrame):
    if (len(pointsSlide) == 0 or len(pointsSlide) == 0):
        return 2
    centroidSlides = computeCentroid(pointsSlide)
    centroidFrame = computeCentroid(pointsFrame)
    normalizedSlides = (centroidSlides[0]/dimSlides[0], centroidSlides[1]/dimSlides[1])
    normalizedFrame = (centroidFrame[0]/dimFrame[0], centroidFrame[1]/dimFrame[1])
    ##     return distance(normalizedSlides, normalizedFrame) / len(pointsSlide)
    return distance(normalizedSlides, normalizedFrame) * (1/len(pointsSlide))

## compute centroid of points
def computeCentroid(listOfXY):
    xAvg = sum([x[0] for x in listOfXY])/len(listOfXY)
    yAvg = sum([x[1] for x in listOfXY])/len(listOfXY)
    return (xAvg, yAvg)


def drawMatches(templateImageKP, videoImageKP, templateImg, videoImage, good):
    MIN_MATCH_COUNT = 10

    if len(good)>10:
        src_pts = np.float32([ templateImageKP[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ videoImageKP[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = templateImg.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        videoImage = cv2.polylines(videoImage,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)

    img3 = cv2.drawMatches(templateImg,templateImageKP,videoImage,videoImageKP,good,None,**draw_params)

    plt.imshow(img3, 'gray'),plt.show()
    return 0


## Usage: cvSIFTFN_save.py path/video/frame.png path/slide/frame.png path/save/file.png
import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys

MIN_MATCH_COUNT = 10


def saveMatchingPaths(slidePath, videoPath, savePath):
    ## IMG
    videoImage = cv2.imread(videoPath, 0)
    template = cv2.imread(slidePath, 0)

    surfDetector = cv2.xfeatures2d.SIFT_create()

    videoImageKP, desV = surfDetector.detectAndCompute(videoImage, None)
    templateImageKP, desT = surfDetector.detectAndCompute(template, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desT,desV,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    print(len(good))

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ templateImageKP[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ videoImageKP[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = template.shape
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

    img3 = cv2.drawMatches(template,templateImageKP,videoImage,videoImageKP,good,None,**draw_params)

    plt.imsave(savePath, img3)

## plt.imshow(img3, 'gray')
## plt.show()


def saveMatchingImgs(slide, video, savePath):
    ## IMG
    videoImage = video
    template = slide

    surfDetector = cv2.xfeatures2d.SIFT_create()

    videoImageKP, desV = surfDetector.detectAndCompute(videoImage, None)
    templateImageKP, desT = surfDetector.detectAndCompute(template, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)


    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desT,desV,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    print(len(good))

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ templateImageKP[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ videoImageKP[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = template.shape
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

    img3 = cv2.drawMatches(template,templateImageKP,videoImage,videoImageKP,good,None,**draw_params)

    plt.imsave(savePath + "_matchNum_" + str(len(good)) + ".png", img3)

    return len(good)


def saveMatchingPaths(slidePath, videoPath, savePath):
    ## IMG
    videoImage = cv2.imread(videoPath, 0)
    template = cv2.imread(slidePath, 0)

    surfDetector = cv2.xfeatures2d.SIFT_create()

    videoImageKP, desV = surfDetector.detectAndCompute(videoImage, None)
    templateImageKP, desT = surfDetector.detectAndCompute(template, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)


    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desT,desV,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    print(len(good))

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ templateImageKP[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ videoImageKP[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = template.shape
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

    img3 = cv2.drawMatches(template,templateImageKP,videoImage,videoImageKP,good,None,**draw_params)

    plt.imsave(savePath, img3)

## plt.imshow(img3, 'gray')
## plt.show()


def frameNumImgs(slide, video):
    ## IMG
    videoImage = video
    template = slide

    surfDetector = cv2.xfeatures2d.SIFT_create()

    videoImageKP, desV = surfDetector.detectAndCompute(videoImage, None)
    templateImageKP, desT = surfDetector.detectAndCompute(template, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)


    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desT,desV,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    print(len(good))

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ templateImageKP[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ videoImageKP[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = template.shape
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

    img3 = cv2.drawMatches(template,templateImageKP,videoImage,videoImageKP,good,None,**draw_params)


    return len(good), img3
## plt.imshow(img3, 'gray')
## plt.show()


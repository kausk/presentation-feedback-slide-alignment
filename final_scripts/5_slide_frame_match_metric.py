import cv2
import os
import pickle
from utilities import visualize, drawMatches, saveMatches, predictARGMIN
from utilities import calculateKeypoints, serializableKeypts, slidesInRange, convertPickledToKPDesc, computeRDistanceDifference, computeSumDiffKeypoints
import numpy as np
import json
import sys







def compute_match_save(SLIDES_PATH, SLIDE_START, SLIDE_END, FRAME_SAMPLE_SAVE_FOLDER, SLIDE_KEYPTS_DESCS_PKL_SAVE_PATH, VID_START, VID_END, VID_KEYPTS_DESCS_PKL_SAVE_PATH, FINAL_FIGURES_RESULTS_SAVE_PATH):

    NUM_SLIDES = SLIDE_END - SLIDE_START
    NUM_FRAMES = VID_END - VID_START
    ## getting filenames
    videoFrameNames = [ filename for filename in os.listdir(FRAME_SAMPLE_SAVE_FOLDER) if filename.endswith(".png") ]
    videoFrameNames = slidesInRange(videoFrameNames, VID_START, VID_END)

    slideNames = [ filename for filename in os.listdir( SLIDES_PATH) if filename.endswith(".png")] 
    slideNames = slidesInRange(slideNames, SLIDE_START, SLIDE_END)





    sampleImgSlides = cv2.imread(slideImgPath + slideNames[0], 0)
    sampleVidSlides = cv2.imread(vidFramePath + videoFrameNames[0], 0)


    ## slide and video dimensions
    slideDimensions = (sampleImgSlides.shape[0], sampleImgSlides.shape[1])
    vidDimensions = (sampleVidSlides.shape[0], sampleVidSlides.shape[1])

    ## load dictionary of things
    slideImgKeypts = pickle.load( open( SLIDE_KEYPTS_DESCS_PKL_SAVE_PATH, "rb" ) )
    vidImgKeypts = pickle.load( open( VID_KEYPTS_DESCS_PKL_SAVE_PATH,  "rb" ) )

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

    ## cross matching
    matchID = 0
    for slide in range(0, NUM_SLIDES):
        for frame in range(0, NUM_FRAMES):
            print("Slide, Frame", (slide, frame), slideNames[slide], videoFrameNames[frame])
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
            slideImg = cv2.imread(slideImgPath + slideNames[slide], 0)  
            vidImg = cv2.imread(vidFramePath + videoFrameNames[frame], 0)
            saveMatches(slidekpts, framekpts, slideImg, vidImg, good, savePath + str(slide) + '-' + str(frame) + '-match.jpg')
            '''

            matchMetric[(slide, frame)] = computeSumDiffKeypoints(slideKeypoints, frameKeypoints, slideDimensions, vidDimensions)
            matchMetricNotNormalized[(slide, frame)] = computeSumDiffKeypoints(slideKeypoints, frameKeypoints, slideDimensions, vidDimensions, False)
    
    ## save visualizations
    mat = visualize(matchMetric, NUM_SLIDES, NUM_FRAMES, FINAL_FIGURES_RESULTS_SAVE_PATH + "normalized_sumDistanceKeypoints.jpg")
    visualize(matchMetricNotNormalized, NUM_SLIDES, NUM_FRAMES, FINAL_FIGURES_RESULTS_SAVE_PATH + "sumDistanceKeypoints.jpg")

    ## save matrix as txt
    np.savetxt(FINAL_FIGURES_RESULTS_SAVE_PATH + "slides_by_vidframes_metric.txt", mat, fmt='%d')

    col_best_prediction = {}
    for col in range(NUM_FRAMES):
        slides = list(mat[:,col])
        best = slides.index(min(slides))
        col_best_prediction[col] = best

    print(json.dumps(col_best_prediction, sort_keys=True))
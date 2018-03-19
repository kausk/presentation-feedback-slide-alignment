import cv2
import pickle
import os
from utilities import calculateKeypoints, serializableKeypts, slidesInRange

VID_START = 0
VID_END = 22
SLIDE_START = 0
SLIDE_END = 12
NUM_SLIDES = VID_END - VID_START - 1

## where to load/save
slideImgPath = "./new_slides/"
vidFramePath = "./new_frames_projected/"

slideImgKeyptsPath = "./new_slide_keypts/"
vidImgKeyptsPath = "./new_vid_keypts/"






def save_descriptors_keypoints(SLIDES_PATH, VID_FRAMES_PATH, SLIDE_START, SLIDE_END, VID_START, VID_END, SLIDES_KP_SAVE_PATH, VID_KP_SAVE_PATH):

    ## getting filenames
    videoFrameNames = [ filename for filename in os.listdir(VID_FRAMES_PATH) if filename.endswith(".png") ]
    videoFrameNames = slidesInRange(videoFrameNames, VID_START, VID_END)

    slideNames = [ filename for filename in os.listdir( SLIDES_PATH) if filename.endswith(".png")] 
    slideNames = slidesInRange(slideNames, SLIDE_START, SLIDE_END)

    ## calculating, saving keypoints
    slideNameKPDesc = dict()
    for slideName in slideNames:
        path = SLIDES_PATH + slideName
        img = cv2.imread(SLIDES_PATH + slideName, 0)
        keypoints, descriptors = calculateKeypoints(img)
        serialized = serializableKeypts(keypoints, descriptors)
        print("Saving " + path)
        slideNameKPDesc[path] = serialized
    pickleOutput = open(SLIDES_KP_SAVE_PATH + "slideKeyptsDesc.pkl", 'wb')
    pickle.dump(slideNameKPDesc, pickleOutput)


    vidNameKPDesc = dict()
    for vidName in videoFrameNames:
        path = VID_FRAMES_PATH + vidName
        img = cv2.imread(VID_FRAMES_PATH + vidName, 0)
        keypoints, descriptors = calculateKeypoints(img)
        print("Saving " + path)
        serialized = serializableKeypts(keypoints, descriptors)
        vidNameKPDesc[path] = serialized
    pickleOutput = open(VID_KP_SAVE_PATH + "vidKeyptsDesc.pkl", 'wb')
    pickle.dump(vidNameKPDesc, pickleOutput)







        

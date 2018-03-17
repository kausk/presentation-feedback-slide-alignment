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


## getting filenames
videoFrameNames = [ filename for filename in os.listdir(vidFramePath) if filename.endswith(".png") ]
videoFrameNames = slidesInRange(videoFrameNames, VID_START, VID_END)

slideNames = [ filename for filename in os.listdir( slideImgPath) if filename.endswith(".png")] 
slideNames = slidesInRange(slideNames, SLIDE_START, SLIDE_END)


print(videoFrameNames)
print(slideNames)

## calculating, saving keypoints
slideNameKPDesc = dict()
for slideName in slideNames:
    path = slideImgPath + slideName
    img = cv2.imread(slideImgPath + slideName, 0)
    keypoints, descriptors = calculateKeypoints(img)
    serialized = serializableKeypts(keypoints, descriptors)
    print("Saving " + path)
    slideNameKPDesc[path] = serialized
pickleOutput = open(slideImgKeyptsPath + "slideKeyptsDesc.pkl", 'wb')
pickle.dump(slideNameKPDesc, pickleOutput)


vidNameKPDesc = dict()
for vidName in videoFrameNames:
    path = vidFramePath + vidName
    img = cv2.imread(vidFramePath + vidName, 0)
    keypoints, descriptors = calculateKeypoints(img)
    print("Saving " + path)
    serialized = serializableKeypts(keypoints, descriptors)
    vidNameKPDesc[path] = serialized
pickleOutput = open(vidImgKeyptsPath + "vidKeyptsDesc.pkl", 'wb')
pickle.dump(vidNameKPDesc, pickleOutput)







    

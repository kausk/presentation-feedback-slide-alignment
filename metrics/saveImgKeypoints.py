import cv2
import pickle
import os
from utilities import calculateKeypoints, serializableKeypts, slidesInRange



## where to load/save
slideImgPath = "../new_slides/"
vidFramePath = "../new_frames_projected/"

slideImgKeyptsPath = "./new_slide_keypts/"
vidImgKeyptsPath = "./new_vid_keypts/"


## getting filenames
videoFrameNames = [ filename for filename in os.listdir(vidFramePath) if filename.endswith(".jpg") ]
videoFrameNames = slidesInRange(videoFrameNames, 0, 28)

slideNames = [ filename for filename in os.listdir( slideImgPath) if filename.endswith(".png")] 
slideNames = slidesInRange(slideNames, 2, 30)
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

    serialized = serializableKeypts(keypoints, descriptors)
    vidNameKPDesc[path] = serialized
pickleOutput = open(vidImgKeyptsPath + "vidKeyptsDesc.pkl", 'wb')
pickle.dump(vidNameKPDesc, pickleOutput)







    

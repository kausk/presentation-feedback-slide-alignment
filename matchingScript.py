from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
from cvSIFTFunction import matchVideoSlideSIFT
import cv2
import os


SLIDE_FOLDER = "slides"

VIDEO_FOLDER = "shortened_video"

WRITE_MATCH_DIRECTORY = "matched_2"


## Digest PDF to image
images = convert_from_path("./data/GroupMeeting11-09-17.pdf")

## Done
imgCounter = 0
## for each image
for img in images:
    ## save somewhere
    img.save(fp="./" + SLIDE_FOLDER + "/" + str(imgCounter) + "-slide.png")
    imgCounter += 1

'''
## http://www.scikit-video.org/stable/examples/scene.html#sceneexample


## establish bounds on when the video and slides start/end
## bound: the once video and slide matched, (v,s) (v,s) (v,s) until slide (v+n, s+x)
## runtime: O(V)


## array:
## [v_1 ..... v_V]
## s1: [XX, Xy, Xz - - - ]
## s2: []
## s3: []
## s4: []


## video: every 2 seconds subsample
## list of video frames [......] == v...V
## list of corresponding times in the video [t_1, ..., t_n]
## subsample and save to file


## algorithm:

## slideNum = 0
## match slide num with video frame 
## if frameNum > 0 and prev_slide has changed
## move up slideNum
## do matching again

## other way: exhaustive scan, O(V*S)

## save the video here
secondsTotal = 0
videoTitles = ["mackenzie-talk-1.MP4", "mackenzie-talk-2.MP4"]

for vid in videoTitles:
    
    print("video")

    success = True
    seconds = 0
    video = cv2.VideoCapture("data/" + vid)

    while success:
        video.set(cv2.CAP_PROP_POS_MSEC, seconds)
        success, img = video.read()
        if success:
            path =  VIDEO_FOLDER + "/" + str(secondsTotal) + "-vid" + ".jpg"
            print(path)
            x = cv2.imwrite(path, img)
            print(x)
        
        seconds += 10000
        secondsTotal += 10000


'''



## dictVidFrameSlide: [videoFrame: slide]
## vidFrame = 0
## slideNum = 0
## _, MatchVal = match(slide, video[vidFrame])
## for each videoFrame: V:
## ## img, match = match(slideNum, video[videoFrame])
## ## if abs(match - matchVal) > matchVal*0.1: '''' what if they are similar, same number of matches
## ## ## slideNum += 1
## ## else:
## ## ## 

## for each slide: S:
## ## _, MatchVal = match(slide, video[vidFrame])
## ## while vidFrame < len(videoFrames):
## ## ## 
## ## vidFrame += 1


## algorithm: start with (slide0, frame0)
## iterate (slide0, frame1)
## (slide0, frame2)
## each iteration:
## slideNum = 0
## if slideNum == 0: 
## img, 
## prevMatches = x
## else:
## 

## for each videoFrame: V
## 

## ## 
## expected slide num: 0 ... num(slides) == s...S
## videoFrame: (slide1: match, matchImg)






## slides: [slide: (frame1, frame2, ...)]
## videoFrame = 0
## _, MatchVal = match(slide, video[vidFrame])
## mv = MatchVal
## for s in Slides:
## while abs(mv - matchVal) <= matchVal*0.1 and videoFrame < len(videos):
## ## _, mv = match(slide, video[vidFrame])
## ## dict[s].append((mv, vidFrameTime, img))
## ## videoFrame += 1
## ## 
## ## ## ##
## video 
##
## for slide in range(len(slides)):
## ## imgs = [v[2] for v in dict[slides]]
## ## for i in images:
## ## ## i.show()


videoFrameSlidesDict = dict()

videoFrameNames = [ filename for filename in os.listdir(VIDEO_FOLDER) if filename.endswith(".jpg") ]
slideNames = [ filename for filename in os.listdir(SLIDE_FOLDER) if filename.endswith(".png")] 

videoFrameNames.sort()
slideNames.sort()
'''
print(slideNames)
numSlides = len(slideNames)

slidesVideoFramesDict = dict()
videoFrameNum = 0


for slideN in slideNames:
    print(slideN)
    slidesVideoFramesDict[slideN] = list()
    if videoFrameNum < len(videoFrameNames):
        frameName = videoFrameNames[videoFrameNum]

        vidImgPath = VIDEO_FOLDER + "/" + videoFrameNames[videoFrameNum]
        slideImgPath = SLIDE_FOLDER + "/" + slideN

        videoImg = cv2.imread(vidImgPath, 0)
        slideImg = cv2.imread(slideImgPath, 0)

        try:
            matchValPrevFrame, img = matchVideoSlideSIFT(videoImg, slideImg)
            mv = matchValPrevFrame

            while abs(mv - matchValPrevFrame) <= matchValPrevFrame*0.1 and videoFrameNum < len(videoFrameNames):
                ##
                slidesVideoFramesDict[slideN].append((mv, videoFrameNames[videoFrameNum], img))

                saveImgPath = WRITE_MATCH_DIRECTORY + "/" + str(videoFrameNames[videoFrameNum]) + ".png"
                cv2.imwrite(saveImgPath, img)

                matchValPrevFrame = mv

                videoFrameNum += 1
                vidImgPath = VIDEO_FOLDER + "/" + videoFrameNames[videoFrameNum]
                videoImg = cv2.imread(vidImgPath, 0)
                print(videoFrameNum)
                mv, img = matchVideoSlideSIFT(videoImg, slideImg)
        except:
            print("Exception, moving forward video")
            videoFrameNum += 1

'''

## try the opposite way, iterate through slides, if not move it forward
slideNum = 0
vidImgPath = VIDEO_FOLDER + "/" + videoFrameNames[0]
slideImgPath = SLIDE_FOLDER + "/" + slideNames[slideNum]
videoImg = cv2.imread(vidImgPath, 0)
slideImg = cv2.imread(slideImgPath, 0)

matchValPrevFrame, img = matchVideoSlideSIFT(videoImg, slideImg)
mv = matchValPrevFrame
dictVideoFrameSlide = dict()
for videoFrameN in videoFrameNames:

    
    vidImgPath = VIDEO_FOLDER + "/" + videoFrameN
    slideImgPath = SLIDE_FOLDER + "/" + slideNames[slideNum]
    videoImg = cv2.imread(vidImgPath, 0)
    slideImg = cv2.imread(slideImgPath, 0)

    mv, img = matchVideoSlideSIFT(videoImg, slideImg)

    saveImgPath = WRITE_MATCH_DIRECTORY + "/" + str(videoFrameN) + ".png"

    if slideNum < len(slideNames) and abs(mv - matchValPrevFrame) <= matchValPrevFrame*0.1:
        print("success")

        cv2.imwrite(saveImgPath, img)

        matchValPrevFrame = mv
            
    else:
        print("moving forward slide")
        slideNum += 1
        slideImgPath = SLIDE_FOLDER + "/" + slideNames[slideNum]
        slideImg = cv2.imread(slideImgPath, 0)
        matchValPrevFrame, img = matchVideoSlideSIFT(videoImg, slideImg)
        cv2.imwrite(saveImgPath, img)






        


'''
img, matchValuePrevFrame = matchVideoSlideSIFT(videoFrameNames[0], slideNames[0])
mv = matchValuePrevFrame

for slideN in slideNames:
    slidesVideoFramesDict[slideN] = list()

    img, matchValuePrevFrame = matchVideoSlideSIFT(videoFrameNames[videoFrame], slideNames[0])
    mv = matchValuePrevFrame
    vidFrameTime = videoFrameNames[videoFrame] 

    while abs(mv - matchValuePrevFrame) <= matchValuePrevFrame*0.1 and videoFrame < len(videoFrameNames):
        slidesVideoFramesDict[slideN].append((mv, ))
        

        vidImgPath = VIDEO_FOLDER + "/" + videoFrameNames[videoFrame]
        slideImgPath = SLIDE_FOLDER + "/" + slideN

        videoImg = cv2.imread(vidImgPath, 0)
        slideImg = cv2.imread(slideImgPath, 0)

        img, mv = matchVideoSlideSIFT(videoImg, slideImg)




slideNum = 0
for vidFrame in videoFrameNames:
    for slide in range(min(0, slideNum-1), min(slideNum+1, numSlides) ):
        vidImgPath = VIDEO_FOLDER + "/" + vidFrame
        slideImgPath = SLIDE_FOLDER + "/" + slideNames[slide]

        print(vidImgPath)
        print(slideImgPath)

        videoImg = cv2.imread(vidImgPath, 0)
        slideImg = cv2.imread(slideImgPath, 0)

        matches, matchedImg = matchVideoSlideSIFT(videoImg, slideImg)
        print(matches)


## for v: VidaoFrames:
## ## for slides: (num-2, num+2)
## ## ## go through num-2, num+2, find the max
## ## ## draw, save the matching for the max
## ## ## save the max_slide in the array
## ## ## opportunity for error: what if mapping is onto something else

## afterwards:
## for each video frame:
## ## ## assign slide with max-match
## ## ##

## todo: too low threshold, human has to do manually
## todo: if unidentifiable, give the top 3 suggestions
## for each v_x: match the highest corresponding s_y
'''
import cv2

BB_VIDEO_PATH = "./original_videos/V0210022.mp4"
BB_VIDEO_SAVE_FOLDER = "./newdata_vid_frames_sampled_fullframe"

sampleAtSeconds = [0, 4000, 6000, 7000, 10000, 13000, 15000, 18000, 22000, 26000, 27000, 30000 ,31000, 34000, 36000, 37000, 39000, 42000, 43000, 45000, 47000, 51000, 56000]
video = cv2.VideoCapture(BB_VIDEO_PATH)

videoFrames = []
videoFramesDict = dict()
index = 0
for secSnapshot in sampleAtSeconds:
    video.set(cv2.CAP_PROP_POS_MSEC, secSnapshot)
    success, img = video.read()
    print(success)
    if success:
        path = BB_VIDEO_SAVE_FOLDER + "/png/" + str(index) + "-" +str(secSnapshot) + ".png"
        print(path)
        x = cv2.imwrite(path, img)
        print(x)
        videoFrames.append(img)
        videoFramesDict[index] = img
        index += 1
    else:
        print("not a success")
print("slide images", len(videoFrames))
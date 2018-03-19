import cv2




def sample_save_video_frames(BB_VIDEO_PATH, BB_VIDEO_SAVE_FOLDER, seconds_start, seconds_end):
    
    video = cv2.VideoCapture(BB_VIDEO_PATH)

    sampleAtSeconds = list(range(seconds_start, seconds_end))
    index = 0
    for secSnapshot in sampleAtSeconds:
        video.set(cv2.CAP_PROP_POS_MSEC, secSnapshot)
        success, img = video.read()
        if success:
            path = BB_VIDEO_SAVE_FOLDER + str(index) + "-" +str(secSnapshot) + ".png"
            x = cv2.imwrite(path, img)
            index += 1
        else:
            print("not a success")

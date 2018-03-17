from PIL import Image
import os

ORIG_IMG_PATH = "/Users/ksk/img/metrics_newdata/vid_frames_sampled_fullframe/"


SAVE_IMG_PATH = "/Users/ksk/img/metrics_newdata/vid_frames_sampled_fullframe/png/"

## read all images
videoFrameNames = [ filename for filename in os.listdir(ORIG_IMG_PATH) if filename.endswith(".jpg") ]


## save as png
for v in videoFrameNames:
    im = Image.open(ORIG_IMG_PATH + v)
    newName = v[:-4] + ".png"
    im.save(SAVE_IMG_PATH + newName)
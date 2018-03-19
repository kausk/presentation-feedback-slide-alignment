import cv2
import numpy as np


## from https://stackoverflow.com/questions/38665277/crop-half-of-an-image-in-opencv
def crop_left_half(image):
    shape = image.shape
    cropped_img = image[0:int(shape[0]),   0:int(shape[1]/2)]
    return cropped_img

def crop_right_half(image):
    shape = image.shape
    cropped_img = image[0:int(shape[0]),  int(shape[1]/2):int(shape[1])]
    return cropped_img

def combine_right_left(right_img, left_img):
    return right_img + left_img

def rearrange(img, save_path):
    right_half = crop_right_half(img)

    left_half = crop_left_half(img)

    vis = np.concatenate((right_half, left_half), axis=1)


    cv2.imwrite(save_path,vis)


def rearrange_all_in_path(vid_frames_path):
    videoFrameNames = [ filename for filename in os.listdir(vidFramePath) if filename.endswith(".png") ]
    for v in videoFrameNames:
        new_img = rearrange(vid_frames_path + v, center_lat_lon, frame_src_corners, slide_dst_corners, example_slide)
        cv2.imwrite(vid_frames_path + v, new_img)



## first test on one image



## create a function: given img, output save location, rearranges and saves it there

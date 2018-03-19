from save_homography import transform_all_in_path
from split_rearrange_img import rearrange_all_in_path
from save_pdf_as_imgs import save_pdf_as_pngs
from 1_sample_vid_frames import sample_save_video_frames
from 4_save_img_keypts_descriptors import save_descriptors_keypoints
from 5_slide_frame_match_metric import compute_match_save
 
## Run all scripts to obtain a slide match prediction for each video


### --- Existing data --- ###
## Where slides are saved, and which slides are in video
SLIDES_PATH = "./360-video-10-23/"
SLIDES_OF_INTEREST_START = 0
SLIDES_OF_INTEREST_END = 63
## Where video is saved, and which seconds are of interest
VIDEO_PATH = "./subsampled_slides_video/cropped_shortened_pres.mp4"
VIDEO_START_SEC = 0
VIDEO_END_SEC = 63


### --- Save Locations of intermediate, final data --- ###
BREAKUP_PDF = False
PDF_PATH = ""
SLIDES_PATH = "" ## where you want the slides to be saved

## Save sampled video frames
FRAME_SAMPLE_SAVE_FOLDER = "" 
## Where to save intermediate pickle file of keypoints/descs
SLIDE_KEYPTS_DESCS_PKL_SAVE_PATH = "./frames_slides_matches/"
VID_KEYPTS_DESCS_PKL_SAVE_PATH = "./vid_frames_sampled/"
## Where to save final results
FINAL_MATCH_MATRIX_SAVE_PATH = ""
FINAL_FIGURES_RESULTS_SAVE_PATH = ""

## Mark whether projection would be needed
## Note: current behavior is to overwrite sampled video frames
PROJECTION = False
CENTER_LAT_LON = (74,185)
SRC_VIDEO_FRAME_CORNERS = [[38,68],[284,83],[298,218],[10,213]] 
DST_SLIDE_CORNERS = [[0,260],[5000,0],[5000,2800],[0,2800]]
EXAMPLE_SLIDE_PATH = ""


## Mark whether slide needs to be split in half, rearranged
SPLIT_REARRANGE = False


## Bagging stuff -- will come later

'''
Full details in README.txt
Data flow process: (slide, video) --> [ [] [] [] ] --> argmax predictions
1. slides_every_second.py: sample, save video frames
2. transformation_script.py: transform video slides if necessary
3. rearrange_all_in_path: maybe need to split, rearrange video
4. save_pdf_as_imgs.py: if slides are one PDF -- break up into multiple PNGs
5. saveImgKeypoints.py: save keypoints, descriptors in pickle file
6. computeDiffKeyptLocations.py: matrix of each [slide1 ... slideM] x [frame1 ... frameN], argmax -- best predictions
7. bagging
8. rerun matrix with bagging
'''

## Step 1 -- sample, save video frames
sample_save_video_frames(VIDEO_PATH, FRAME_SAMPLE_SAVE_FOLDER, VIDEO_START_SEC, VIDEO_END_SEC)

## Step 2 -- if need to split and rearrange, do that
if SPLIT_REARRANGE:
    rearrange_all_in_path(FRAME_SAMPLE_SAVE_FOLDER)

## Step 3 -- if need to transform
if PROJECTION:
    transform_all_in_path(FRAME_SAMPLE_SAVE_FOLDER, CENTER_LAT_LON, SRC_VIDEO_FRAME_CORNERS, DST_SLIDE_CORNERS, EXAMPLE_SLIDE_PATH)

## Step 4 -- save pdf slides as individual pngs
if BREAKUP_PDF:
    save_pdf_as_pngs(PDF_PATH, SLIDES_PATH)


## Step 5 -- Save img and video keypoints and descriptors
save_descriptors_keypoints(SLIDES_PATH, FRAME_SAMPLE_SAVE_FOLDER, 
            SLIDES_OF_INTEREST_START, SLIDES_OF_INTEREST_END, 
            VIDEO_START_SEC, VIDEO_END_SEC, 
            SLIDE_KEYPTS_DESCS_PKL_SAVE_PATH, 
            VID_KEYPTS_DESCS_PKL_SAVE_PATH)
 
## Step 6 -- Compute Most Likely Slide for each Video Frame
compute_match_save(SLIDES_PATH, 
                SLIDES_OF_INTEREST_START, 
                SLIDES_OF_INTEREST_END, 
                FRAME_SAMPLE_SAVE_FOLDER, 
                SLIDE_KEYPTS_DESCS_PKL_SAVE_PATH, 
                VIDEO_START_SEC, 
                VIDEO_END_SEC, 
                VID_KEYPTS_DESCS_PKL_SAVE_PATH, 
                FINAL_FIGURES_RESULTS_SAVE_PATH)


## Will print out the JSON of predictions at the end of compute_match_save, and save the prediction matrix at FINAL_FIGURES_RESULTS_SAVE_PATH folder
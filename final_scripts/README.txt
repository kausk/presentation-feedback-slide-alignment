Files and descriptions of their uses
U: computeDiffKeyptLocations.py
computeXMatchingCentroid.py
computeXMatchingDistanceKeypt.py
convertJPGtoPNG.py
sample_video_frames.py
save_pdf_as_imgs.py
saveAllCrossMatches.py
saveImgKeypoints.py
saveImgs_horizontally.py
slide_bucketing.py
slides_every_second.py
split_rearrange_img.py
transformation_script.py
utilities.py


Data flow process: (slide, video) --> [ [] [] [] ] --> argmax predictions
1. slides_every_second.py: 
sample, save video frames
2. transformation_script.py
transform video slides if necessary
3. save_pdf_as_imgs.py
if slides are one PDF -- break up into multiple PNGs
4. saveImgKeypoints.py 
save keypoints, descriptors in pickle file
5. computeDiffKeyptLocations.py
matrix of each [slide1 ... slideM] x [frame1 ... frameN]
argmax -- best predictions
6. bagging
7. rerun matrix with bagging



Endpoints in each file
-1. saving slide pdfs, video frames
0. possibly having to transform videos
1. saveImgKeypoints.py:
- VID_START, VID_END: start, end seconds of video
- slideImgPath, vidFramePath: where frames, slide imgs are currently located
- slideImgKeyptsPath, vidImgKeyptsPath where to save pickled keypoints, descriptors
2. 


Addition: bag [ [] [] [] [] ] --> [ [1] [1] [2] [2] [3] [3] [3] ]


Addition: each frame: compute distance between all groups instead of all slides, pick argmax


Todo: bag, compute frame --> groups of slides --> region with min distance (weight with number of slides in group)


Todo: accuracy of this method

Todo: accuracy of this method vs. without bagging





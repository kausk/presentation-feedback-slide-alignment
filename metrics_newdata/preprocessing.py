import os
## Rename slides
slideFramePath = "./360-video-10-23/" 
slideNames = [ filename for filename in os.listdir(slideFramePath) if filename.endswith(".jpg") ]
counter = 0

for sn in sorted(slideNames):
    os.rename(slideFramePath + sn, slideFramePath + str(counter) + '-' + sn)
    counter += 1
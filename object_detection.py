from descriptors.orb import ORB
import numpy as np
import os
import cv2

# Append all the target images
img_path = "VisionComputacional/Examenes/Ex02/resources/test" # Save the root path
images = []
classNames = []
testImages = os.listdir(img_path) # List target images of the directory
print("Total class detected: ", len(testImages))

# Loop to read and append the images
for cl in testImages:
    imgCur = cv2.imread(f'{img_path}/{cl}')
    if imgCur is None:
        raise ValueError(f"Cannot load image from {img_path}/{cl}")
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0]) # Stores the classification names

print(classNames) # print classification names

# Append all the train videos
vid_path = "VisionComputacional/Examenes/Ex02/resources/train"
videos = []
trainVideos = os.listdir(vid_path) # List videos of the directory
print("Total videos detected: ", len(trainVideos))

# Loop to open and append videos
for vid in trainVideos:
    vidCur = cv2.VideoCapture(f'{vid_path}/{vid}')
    if not vidCur.isOpened():
        raise ValueError(f"Cannot open video from {vid_path}/{vid}")
    videos.append(vidCur)

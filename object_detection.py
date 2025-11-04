from descriptors.orb import ORB
from descriptors.sift import SIFT
from descriptors.hog_svm import HOG_SVM
import pandas as pd
import os
import cv2


def read_images(src, path_images):
    images = []
    for path in path_images:
        imgCur = cv2.imread(f'{src}/{path}')
        if imgCur is None:
            raise ValueError(f"Cannot load image from {src}/{path}")
        images.append(imgCur)
    return images

# ---------------------- IMAGES FOR ORB AND SIFT ------------------------
img_path = "VisionComputacional/Examenes/Ex02/resources/train" # Save the root path
images = []
classNames = ["calendar", "notebook", "raid"]
trainImages = os.listdir(img_path)
trainImages.pop(1)

print(trainImages)
print("Total class detected: ", len(trainImages))

images = read_images(img_path, trainImages)

# -------------------- IMAGES FOR TRAINING SVM ------------------------
img_path = "VisionComputacional/Examenes/Ex02/resources/train/hog_img"

hog_paths = os.listdir("VisionComputacional/Examenes/Ex02/resources/train/hog_img")
print(hog_paths)
hog_images = []
hog_labels = ["calendar", "calendar", "calendar","calendar",  "notebook", "notebook", "notebook", "notebook", "raid", "raid", "raid", "raid", "raid"]

hog_images = read_images(img_path, hog_paths)

# Append all the test videos
vid_path = "VisionComputacional/Examenes/Ex02/resources/test"
videos = os.listdir(vid_path) # List videos of the directory
print("Total videos detected: ", len(videos))

# -----------------------------  ORB descriptor  -------------------------------
orb_detector = ORB(images, classNames) # Create ORB descriptor
#orb_detector.plot_comparison() # Show the keypoints of all the images

# ----------------------------  SURF descriptor  -------------------------------
sift_detector = SIFT(images, classNames) # Create SURF descriptor
#sift_detector.plot_comparison() # Show the keypoints of all the images

# ----------------------------- HOG descriptor ---------------------------------
# Train model
modelo = HOG_SVM(hog_images, hog_labels)
modelo.train_SVM()

# ----------------------------- OBJECT DETECTION -------------------------------
metrics = []

# Open video
vid_path_full = "VisionComputacional/Examenes/Ex02/resources/test/video.mp4"
vidCur = cv2.VideoCapture(vid_path_full)
if not vidCur.isOpened():
    raise ValueError(vid_path_full)

# Get ORB metrics
orb_metrics = orb_detector.compare_keypoints(vidCur)
metrics.append(orb_metrics)

# Reopen the video for SIFT
vidCur.release()
vidCur = cv2.VideoCapture(vid_path_full)

# Get SIFT metrics
sift_metrics = sift_detector.compare_keypoints(vidCur)
metrics.append(sift_metrics)

# Reopen the video for HOG + SVM
vidCur.release()
vidCur = cv2.VideoCapture(vid_path_full)

# Get HOG + SVM metrics
hog_metrics = modelo.classify_video(vidCur)
metrics.append(hog_metrics)

vidCur.release()

print(pd.DataFrame(metrics))
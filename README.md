# Object Identification Using ORB, SIFT, and HOG Descriptors in OpenCV

## Problem Statement

The objective of this project is to implement an object identification system using feature descriptors available in OpenCV. The system must recognize selected objects from either a live video stream or a new input image.

The following feature descriptors were implemented and compared:

- ORB (Oriented FAST and Rotated BRIEF)
- SIFT (Scale-Invariant Feature Transform)
- HOG (Histogram of Oriented Gradients) + SVM classifier

The selected objects for identification were:

- Notebook
- Raid insecticide box
- Calendar

Training images are located in `resources/train/`, and evaluation was performed using a prerecorded video in `resources/test/`.

---

## Methodology Overview

Each descriptor was implemented in an independent class (located in `descriptors/`), where target object features were computed during initialization.

### ORB

ORB combines the FAST keypoint detector with the BRIEF descriptor.

- Keypoints detected using FAST
- Keypoints ranked using Harris corner response
- Binary descriptors generated using oriented BRIEF
- Feature matching performed using BFMatcher with KNN
- Lowe’s ratio test (0.75) applied to filter ambiguous matches
- Object classified based on the highest number of good matches
- Detection threshold: 13 matches

ORB is computationally efficient and suitable for real-time processing but is sensitive to motion and viewpoint changes.

---

### SIFT

SIFT detects and describes keypoints invariant to scale, rotation, and illumination changes.

- Keypoints and descriptors extracted using `detectAndCompute()`
- Feature matching with BFMatcher + KNN
- Lowe’s ratio test applied
- Object selected based on the highest number of valid matches
- Detection threshold: 100 matches

SIFT demonstrated the most robust detection performance among the three descriptors, although it was significantly slower.

---

### HOG + SVM

HOG extracts gradient-based features, and classification is performed using a Support Vector Machine (SVM).

- Total training images: 13  
  - 4 calendar  
  - 4 notebook  
  - 5 raid box  
- Images converted to grayscale, equalized, and resized
- HOG descriptor parameters:
  - Window size: 64×128
  - Block size: 16×16
  - Block stride: 8×8
  - Cell size: 4×4
  - 9 orientation bins
- Labels encoded using LabelEncoder
- SVM hyperparameters optimized using GridSearch
- Classification based on probability threshold (0.4)

Unlike ORB and SIFT, HOG does not rely on feature matching but on supervised learning. It requires more training data for improved performance.

---


## Conclusion

This project demonstrates the practical differences between local feature matching methods (ORB, SIFT) and feature-based classification (HOG + SVM).

- SIFT provided the most accurate and stable object recognition.
- ORB offered faster processing with moderate robustness.
- HOG + SVM required more training data but achieved high processing speed.

OpenCV significantly simplifies descriptor extraction, feature matching, and model integration, making it an effective framework for real-time object identification tasks.

---

## References

[1] O. Günaydın, “Detecting and Tracking Objects with ORB Algorithm using OpenCV,” Medium.  
https://medium.com/thedeephub/detecting-and-tracking-objects-with-orb-using-opencv-d228f4c9054e  

[2] Murtaza's Workshop – Robotics and AI, “Feature Detection and Matching + Image Classifier Project | OpenCV Python,” YouTube, 2020.  
https://www.youtube.com/watch?v=nnH55-zD38I  

[3] OpenCV Documentation, “Introduction to SIFT (Scale-Invariant Feature Transform),” OpenCV-Python Tutorials.  
https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html  

[4] D. Nemutlu, “HOG Feature Descriptor,” Medium.  
https://medium.com/@dnemutlu/hog-feature-descriptor-263313c3b40d  

[5] Ramon Ariel Ivan Muñoz Corona, “Python OpenCV - Detección de Peatones con HOG Descriptor y SVM,” YouTube, 2023.  
https://www.youtube.com/watch?v=Kz5PMCmLrHg
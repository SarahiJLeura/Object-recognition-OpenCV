import cv2
import numpy as np
import time
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import joblib

class HOG_SVM:
    def __init__(self, trainingImgs, classNames):
        self.classNames = classNames
        self.trainingImages = trainingImgs
        self.hog = cv2.HOGDescriptor(
            _winSize=(64, 128),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        self.model = None
        self.label_encoder = LabelEncoder()
        print("[INFO] Initializing HOG + SVM...")

    def compute_descriptors(self):
        descriptors = []
        for img in self.trainingImages:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 128))
            desc = self.hog.compute(resized)
            descriptors.append(desc.flatten())
        return np.array(descriptors, dtype=np.float32)

    def train_SVM(self):
        print("[INFO] Calculating HOG descriptors...")
        X = self.compute_descriptors()
        y = self.label_encoder.fit_transform(self.classNames)

        print("[INFO] Training SVM model...")
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        # Find the best parameters to create a model
        grid = GridSearchCV(
            svm.SVC(probability=True),
            param_grid,
            refit=True,
            cv=3,
            scoring='accuracy'
        )
        grid.fit(X, y) # Fit the model
        self.model = grid.best_estimator_

        print(f"[INFO] Comple training. Best model: {grid.best_params_}")
        joblib.dump((self.model, self.label_encoder), "modelo_HOG_SVM.pkl")

    def load_model(self, path="modelo_HOG_SVM.pkl"):
        self.model, self.label_encoder = joblib.load(path)
        print("[INFO] SVM model successfully uploaded.")

    def predict_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 128))
        desc = self.hog.compute(resized).flatten().reshape(1, -1)
        pred = self.model.predict(desc)[0]
        class_name = self.label_encoder.inverse_transform([pred])[0]
        return class_name

    def classify_video(self, cap):
        t0 = time.time()
        n_frames = 0

        print("[INFO] Press 'ESC' to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the image only for the classifier no to show it
            label = self.predict_frame(frame)

            # Draw in the original image
            cv2.putText(frame, label, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 2)

            cv2.imshow("Detection HOG + SVM", frame)

            n_frames += 1

            if cv2.waitKey(1) & 0xFF == 27: # Wait to ESC to exit
                break

        cap.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - t0
        avg_fps = n_frames / elapsed if elapsed > 0 else 0

        # Final metrics
        metrics = {
            "descriptor": "HOG + SVM",
            "frames": n_frames,
            "avg_fps": avg_fps,
            "avg_keypoints": "X",
            "avg_matches": "X"
        }
        return metrics



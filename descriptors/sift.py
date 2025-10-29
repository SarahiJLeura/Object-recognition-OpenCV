import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

class SIFT:
    def __init__(self, targets, class_names):
        """
        targets: list of target images (numpy arrays, BGR)
        class_names: list of class names corresponding to each target image
        """
        self.targets = targets
        self.class_names = class_names

        # Initialize the SIFT detector
        self.sift = cv2.SIFT_create()

        # Precompute keypoints and descriptors for all target images
        self.keypoints_list = []
        self.descriptors_list = []
        self.rgb_images = []

        for img in self.targets:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.sift.detectAndCompute(gray, None)
            self.keypoints_list.append(kp)
            self.descriptors_list.append(des)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.rgb_images.append(rgb_img)

        print(f"[INFO] Computed {len(self.targets)} SIFT descriptors for target images.")

    def compare_keypoints(self, video, threshold_matches=40):
        """
        Compare features between video frames and target images.
        
        video: cv2.VideoCapture object
        threshold_matches: minimum number of good matches required to recognize an object
        """
        t0 = time.time()
        n_frames = 0
        total_keypoints = 0
        total_matches = 0

        bf = cv2.BFMatcher()

        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp_frame, des_frame = self.sift.detectAndCompute(frame_gray, None)

            if des_frame is None:
                continue  # skip frame if no descriptors were found

            best_match_count = 0
            best_class_name = "Unknowed"

            # Compare the current frame descriptors with each target image
            for des_target, name in zip(self.descriptors_list, self.class_names):
                if des_target is None:
                    continue

                matches = bf.knnMatch(des_target, des_frame, k=2)
                good_matches = []
                for m_n in matches:
                    if len(m_n) < 2:
                        continue
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                if len(good_matches) > best_match_count:
                    best_match_count = len(good_matches)
                    best_class_name = name

            # If not enough matches, label as "Unknowed"
            if best_match_count < threshold_matches:
                best_class_name = "Unknowed"

            # Display the classification on the frame
            cv2.putText(frame, f"Object: {best_class_name}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("SIFT Classification", frame)

            # Collect metrics
            total_matches += best_match_count
            total_keypoints += len(kp_frame)
            n_frames += 1

            if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
                break

        video.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - t0
        avg_fps = n_frames / elapsed if elapsed > 0 else 0

        # Return performance metrics
        metrics = {
            "descriptor": "SIFT",
            "frames": n_frames,
            "avg_fps": avg_fps,
            "avg_keypoints": total_keypoints / max(n_frames, 1),
            "avg_matches": total_matches / max(n_frames, 1)
        }

        return metrics

    def plot_comparison(self):
        """
        Display all target images with their detected keypoints.
        """
        n = len(self.targets)
        cols = 3
        rows = int(np.ceil(n / cols))

        plt.figure(figsize=(15, 5 * rows))
        for i, (rgb_img, kp, name) in enumerate(zip(self.rgb_images, self.keypoints_list, self.class_names)):
            img_kp = cv2.drawKeypoints(rgb_img, kp, None, color=(255, 0, 0), flags=0)
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img_kp)
            plt.title(name)
            plt.axis("off")

        plt.tight_layout()
        plt.show()

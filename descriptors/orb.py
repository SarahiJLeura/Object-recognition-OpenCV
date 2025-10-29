import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

class ORB:
    def __init__(self, targets, class_names):
        """
        targets: Target images (numpy arrays)
        class_names: Array with the classification names
        video: object cv2.VideoCapture
        """
        self.targets = targets
        self.class_names = class_names

        # Initialize ORB detector
        self.orb = cv2.ORB_create()

        # Precalculate keypoints and descriptors of all the target images
        self.descriptors_list = []
        self.keypoints_list = []
        self.rgb_images = []

        for img in self.targets:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(gray, None)
            self.keypoints_list.append(kp)
            self.descriptors_list.append(des)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.rgb_images.append(rgb_img)

        print(f"[INFO] Calculados {len(self.targets)} descriptores ORB para las imagenes objetivo.")

    def compare_keypoints(self, video):
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
            kp_frame, des_frame = self.orb.detectAndCompute(frame_gray, None)

            if des_frame is None:
                continue

            # classification by similarity
            best_match_count = 0
            best_class_name = "Unknowed"

            for des, name in zip(self.descriptors_list, self.class_names):
                if des is None or des_frame is None:
                    continue

                matches = bf.knnMatch(des, des_frame, k=2)
                good_matches = []
                for m_n in matches:
                    if len(m_n) < 2:
                        continue
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                # Find the highest matches
                if len(good_matches) > best_match_count:
                    best_match_count = len(good_matches)
                    best_class_name = name

            # Decide if the object is recognized
            THRESHOLD_MATCHES = 13 
            if best_match_count < THRESHOLD_MATCHES:
                best_class_name = "Unknowed"

            # Show text on screen
            cv2.putText(frame, f"Objeto: {best_class_name}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("ORB Classification", frame)

            # Accumulation of metrics
            total_matches += best_match_count
            total_keypoints += len(kp_frame)
            n_frames += 1

            if cv2.waitKey(1) & 0xFF == 27:  # Esc para salir
                break

        video.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - t0
        avg_fps = n_frames / elapsed if elapsed > 0 else 0

        # Final metrics
        metrics = {
            "descriptor": "ORB",
            "frames": n_frames,
            "avg_fps": avg_fps,
            "avg_keypoints": total_keypoints / max(n_frames, 1),
            "avg_matches": total_matches / max(n_frames, 1)
        }

        return metrics
    

    def plot_comparison(self):
        """Display all the target images with their keypoints"""
        n = len(self.targets)
        cols = 3
        rows = int(np.ceil(n / cols))

        plt.figure(figsize=(15, 5 * rows))
        for i, (rgb_img, kp, name) in enumerate(zip(self.rgb_images, self.keypoints_list, self.class_names)):
            img_kp = cv2.drawKeypoints(rgb_img, kp, None, color=(0, 255, 0), flags=0)
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img_kp)
            plt.title(name)
            plt.axis("off")

        plt.tight_layout()
        plt.show()


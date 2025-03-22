import pickle
import cv2
import numpy as np
import sys
sys.path.append("../")
from utils import measure_distance, measure_xy_distance
import os
class CameraMovementEstimator:
    def __init__(self, frame):
        self.minimum_distance = 5

        self.lk_params = dict(winSize=(15, 15),
                            maxLevel=2,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:105] = 1

        self.features = dict(
                            maxCorners=100,
                            qualityLevel=0.3,
                            minDistance=3,
                            blockSize=7,
                            mask=mask_features)
        


    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read stub if available
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)
        

        camera_movement = [[0,0]]*len(frames)
        
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)

        for frame_number in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_number], cv2.COLOR_BGR2GRAY)
            new_features, status, error = cv2.calcOpticalFlowPyrLK(old_gray,
                                                                frame_gray,
                                                                old_features, None,
                                                                **self.lk_params)
            max_distane = 0
            camera_movement_x , camera_movement_y = 0,0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)

                if distance > max_distane:
                    max_distane = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(new_features_point, old_features_point)
            if max_distane < self.minimum_distance:
                camera_movement[frame_number] = camera_movement_x, camera_movement_y
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            
            old_gray = frame_gray.copy()
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frame = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()  
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]

            frame = cv2.putText(frame, f"X Movement: {x_movement: .2f}", (10, 30),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, f"Y Movement: {y_movement: .2f}", (10, 70),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            output_frame.append(frame)

        
        return output_frame
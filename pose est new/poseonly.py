import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

model_path = ("pose_landmarker_full.task")
video_path = 'muaythaivskarate.mp4'

cap = cv2.VideoCapture(0)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Define the skeleton connections (pairs of landmark indices)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Arms
    (11, 23), (12, 24), (23, 24),                   # Torso
    (23, 25), (24, 26), (25, 27), (26, 28),         # Legs
    (27, 31), (28, 32), (27, 29), (28, 30)          # Feet
]

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    #delegate=python.BaseOptions.Delegate.GPU,
    
    num_poses=6) # Set this to the maximum number of people expected

with PoseLandmarker.create_from_options(options) as landmarker:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        frame_timestamp_ms = int(1000 * frame_index / fps)
        frame_index += 1

        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        black_background = np.zeros(frame.shape, dtype=np.uint8)

        # Process each detected person
        if pose_landmarker_result.pose_landmarks:
            for skeleton in pose_landmarker_result.pose_landmarks:
                
                # 1. Draw the Lines (Connections)
                for connection in POSE_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    # Ensure indices exist in the detection
                    if start_idx < len(skeleton) and end_idx < len(skeleton):
                        pt1 = (int(skeleton[start_idx].x * frame.shape[1]), 
                               int(skeleton[start_idx].y * frame.shape[0]))
                        pt2 = (int(skeleton[end_idx].x * frame.shape[1]), 
                               int(skeleton[end_idx].y * frame.shape[0]))
                        cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
                        #replace frame with black_background
                # 2. Draw the Keypoints
                for landmark in skeleton:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                    #replace frame with black_background

        cv2.imshow('Pose Detection', frame)
                                                  #replace frame with black_background
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
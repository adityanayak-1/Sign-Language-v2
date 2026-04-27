#1. Importing libraries
import cv2
import numpy as np
import os 
from matplotlib import pyplot as plt
import time
import mediapipe as mp
  
# Global variables
mp_holistic= mp.solutions.holistic
mp_drawing= mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable= False
    results= model.process(image)
    image.flags.writeable= True
    image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Pose landmark indices: shoulders(11,12), elbows(13,14), wrists(15,16)
POSE_KEEP = [11, 12, 13, 14, 15, 16]

# Connections between the 6 kept pose landmarks only
POSE_KEEP_CONNECTIONS = [
    (11, 13), (13, 15),  # left shoulder -> elbow -> wrist
    (12, 14), (14, 16),  # right shoulder -> elbow -> wrist
    (11, 12),            # shoulder to shoulder
]

def draw_styled_landmarks(image, results):
    h, w = image.shape[:2]

    # Draw only the 6 selected pose landmarks and their connections
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Lines first (underneath dots)
        for (a, b) in POSE_KEEP_CONNECTIONS:
            ax, ay = int(lm[a].x * w), int(lm[a].y * h)
            bx, by = int(lm[b].x * w), int(lm[b].y * h)
            cv2.line(image, (ax, ay), (bx, by), (80, 44, 121), 2)

        # Dots
        for i in POSE_KEEP:
            cx, cy = int(lm[i].x * w), int(lm[i].y * h)
            cv2.circle(image, (cx, cy), 4, (80, 22, 10), -1)

    # Hands — same colours as before
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[results.pose_landmarks.landmark[i].x,
                      results.pose_landmarks.landmark[i].y,
                      results.pose_landmarks.landmark[i].z,
                      results.pose_landmarks.landmark[i].visibility]
                     for i in POSE_KEEP]).flatten() if results.pose_landmarks else np.zeros(24)
    lh = np.array([(res.x, res.y, res.z) for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([(res.x, res.y, res.z) for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

if __name__ == "__main__":
    cap= cv2.VideoCapture(1)
    with mp_holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence= 0.5) as holistic:
        while cap.isOpened():
            ret, frame= cap.read()
            image, results= mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
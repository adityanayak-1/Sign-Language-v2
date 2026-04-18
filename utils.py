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
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius= 4),
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius= 2))
    
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius= 4),
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius= 2))
    
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius= 4),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius= 2))

# Pose landmark indices: shoulders(11,12), elbows(13,14), wrists(15,16)
POSE_KEEP = [11, 12, 13, 14, 15, 16]

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
    cap= cv2.VideoCapture(0)
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
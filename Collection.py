import matplotlib.pyplot as plt
import math
import mediapipe as mp 
import numpy as np 
import cv2  as cv
import os

image = cv.imread("DATASET/Downdog/00000000.jpg")
mp_pose = mp.solutions.pose #importing pose estimation model from mp
pose = mp_pose.Pose(static_image_mode = True, min_detection_confidence=0.5, model_complexity=2)

plt.figure(figsize = [10,10])

plt.title("Sample Image");plt.axis('off');plt.imshow(image[:,:,::-1]);plt.show()
#Convert to RGB and Perform POSE DETECTION
results = pose.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
# if results.pose_landmarks:
#     for i in range(2):
#         print(f'{mp_pose.PoseLandmark(i).name}:\n{results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value]}')

image_height, image_width, _ = image.shape
if results.pose_landmarks: #Checks for landmarks again
    for i in range(2):
        print(f'{mp}')
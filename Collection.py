import matplotlib.pyplot as plt
import math
import mediapipe as mp 
import numpy as np 
import cv2  as cv
import os
mp_drawing = mp.solutions.drawing_utils #drawing utility, visualition of poses
mp_pose = mp.solutions.pose #importing pose estimation model from mp
image = cv.imread("DATASET/Goddess/00000000.jpg")
def calculate_angle(a, b, c):
    x1, y1, _ = a
    x2, y2, _ = b
    x3, y3, _ = c
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle
 
def classifyPose(landmarks, output_image, display=False):
    label = 'Unknown Label'
    color = (0,0, 255)
    
    left_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    right_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    left_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    right_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
    left_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    right_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    
def pose_checker(img, pose, display = True):
    
	
	image_copy = image.copy()
	pose = mp_pose.Pose(static_image_mode = True, min_detection_confidence=0.5, model_complexity=2)
	lndmrks = []
	image_height, image_width, _ = image.shape
	#Convert to RGB and Perform POSE DETECTION
	results = pose.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
 
	if results.pose_landmarks: #Appending Landmarks into lndmrks
		mp_drawing.draw_landmarks(image_copy, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) #Drawing Landmarks
		for landmark in results.pose_landmarks.landmark:
			lndmrks.append(int(landmark.x * image_width), int(landmark.y * image_height), int(landmark.z * image_width))
        # fig = plt.figure(figsize = [10,10])
		# plt.title("Output");plt.axis('off');plt.imshow(image_copy[:,:,::-1]);plt.show()

	if results.pose_landmarks: #Checks for landmarks 
		for i in range(2): #Display landmarks coordinates on the basis of location
			print(f'{mp_pose.PoseLandmark(i).name}:')
			print(f'x: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x * image_width}')
			print(f'y: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y * image_height}')
			print(f'z: {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z * image_width}')
			print(f'visibility : {results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility}\n')
   
   
   
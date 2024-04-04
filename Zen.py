import mediapipe as mp 
import numpy as np 
import cv2  as cv
import math


mp_drawing = mp.solutions.drawing_utils #drawing utility, visualition of poses
mp_pose = mp.solutions.pose #importing pose estimation model from mp


def calculateAngle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle < 0:
        angle += 360
        
    return angle 

# def calculateAngle(landmark1, landmark2, landmark3):

#     # Get the required landmarks coordinates.
#     x1, y1 = landmark1
#     x2, y2 = landmark2
#     x3, y3 = landmark3
 
#     # Calculate the angle between the three points
#     angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
#     # Check if the angle is less than zero.
#     if angle < 0:
 
#         # Add 360 to the found angle.
#         angle += 360
    
#     # Return the calculated angle.
#     return angle


cap = cv.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #Recolor Image
        image.flags.writeable = False
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv.cvtColor(frame, cv.COLOR_RGB2BGR) #Recoloring it back to BGR
        
        try:
            
            landmarks = results.pose_landmarks.landmark
            #COLLECTING LEFT SIDE INFORMATION AND DISPLAYING
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            #LEFT ELBOW
            le = calculateAngle(left_shoulder, left_elbow, left_wrist)
            cv.putText(image, str(le), tuple(np.multiply(left_elbow, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
            #LEFT WRIST
            lw = calculateAngle(left_elbow, left_wrist, left_shoulder)
            cv.putText(image, str(lw), tuple(np.multiply(left_wrist, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
            #LEFT SHOULDER
            ls = calculateAngle(left_elbow, left_shoulder, left_wrist)
            cv.putText(image, str(ls), tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
            #LEFT HIP
            lh = calculateAngle(left_ankle, left_hip, left_knee)
            cv.putText(image, str(lh), tuple(np.multiply(left_hip, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
            #LEFT KNEE
            lk = calculateAngle(left_hip, left_knee, left_ankle)
            cv.putText(image, str(lk), tuple(np.multiply(left_knee, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
            #LEFT ANKLE
            la = calculateAngle(left_knee, left_ankle, left_hip)
            cv.putText(image, str(la), tuple(np.multiply(left_ankle, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
            
            #COLLECTING RIGHT SIDE INFORMATION AND DISPLAYING
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            #RIGHT ELBOW
            re = calculateAngle(right_shoulder, right_elbow, right_wrist)
            cv.putText(image, str(re), tuple(np.multiply(right_elbow, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
            #RIGHT WRIST
            rw = calculateAngle(right_elbow, right_wrist, right_shoulder)
            cv.putText(image, str(rw), tuple(np.multiply(right_wrist, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
            #RIGHT SHOULDER
            rs = calculateAngle(right_elbow, right_shoulder, right_wrist)
            cv.putText(image, str(rs), tuple(np.multiply(right_shoulder, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
            #RIGHT HIP
            rh = calculateAngle(right_ankle, right_hip, right_knee)
            cv.putText(image, str(rh), tuple(np.multiply(right_hip, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
            #RIGHT KNEE
            rk = calculateAngle(right_hip, right_knee, right_ankle)
            cv.putText(image, str(rk), tuple(np.multiply(right_knee, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
            #RIGHT ANKLE
            ra = calculateAngle(right_knee, right_ankle, right_hip)
            cv.putText(image, str(ra), tuple(np.multiply(right_ankle, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
        
        
        except:
            pass
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) )
        
        
        #print(angle)
        # landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
        # landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        # landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        cv.imshow("Yoga Detection System", image)
        
        #DETECTION
        ## TREE POSE ###
        #Check if one leg is straight
        if lk > 165 and lk < 195 or rk > 165 and rk < 195:
 
        # Check if the other leg is bended at the required angle.
            if lk > 50 and lk < 90 or rk > 50 and rk < 90:
 
            # Specify the label of the pose that is tree pose.
                print("Tree Pose")
                label = 'Tree Pose'
        
        # ## T POSE ##
        if le > 165 and le < 195 and re < 165 and re < 195:
 
        # Check if shoulders are at the required angle.
            if ls > 0 and ls < 20 and rs > 0 and rs > 20 :
                         if lk > 160 and lk < 195 and rk > 160 and rk < 195:
                             print("T Pose")
                     

        ### WARRIOR II POSE ###
        if le > 165 and le < 195 and re > 165 and re < 195:
        # Check if shoulders are at the required angle.
            if ls > 0 and ls < 10 and rs > 0 and rs < 10:
            # Check if one leg is straight.
                if lk > 165 and lk < 195 or rk > 165 and rk < 195:
                # Check if the other leg is bended at the required angle.
                    if lk > 90 and lk < 120 or rk > 90 and rk < 120:
                        print("Warrior II Pose")
                        label = 'Warrior II Pose' 
                
        ### GODDESS POSE ###
        
        if ls > 35 and ls < 70 and rs > 35 and rs < 70:
            if le > 70 and le < 100 and re > 70 and re < 100:
                if lk > 115 and lk < 190 and rk > 115 and rk < 190:
                    print("Goddess Pose")
                    label = "Goddess Pose"
        
        if cv.waitKey(10) & 0xFF == ord('q'): #Checks if Q is pressed to end the stream
            break
    cap.release()
    cv.destroyAllWindows()
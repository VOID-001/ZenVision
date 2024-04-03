import cv2 as cv #Computer Vision
import matplotlib.pyplot as plt #Pyplot for pose detection
capture = cv.VideoCapture(0)
while(True):
    _,frame= capture.read()
    cv.imshow('livestream',frame)

    if cv.waitKey(1) == ord("q"):
        break

capture.release()
cv.destroyAllWindows()
    
#print(cv.__version__)


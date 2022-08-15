import cv2
from object_detector import *
import numpy as np

def camSize(cam):
    #Camera (1280x720)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


parameters = cv2.aruco.DetectorParameters_create()

#Aruco marker (5x5)
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

# Load Object Detector
detector = HomogeneousBgDetector()

# Load Webcam(0 = webcam)
cam = cv2.VideoCapture(0)

while True:
    _, frame = cam.read()

    # Get Aruco marker
    corners, _, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if corners:
        # Vertices must be integers
        int_corners = np.int0(corners)                          

        # Draw a red polygon around the marker(Thickness 5)
        cv2.polylines(frame, int_corners, True, (0, 255, 0), 5)   

        # Aruco marker environment length (pixel)
        aruco_perimeter = cv2.arcLength(corners[0], True)

        # Pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 20

        contours = detector.detect_objects(frame)

        # Draw objects boundaries
        for cnt in contours:
            # Returns the width, length, and angle of the minimum rectangular area surrounding the object
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            # Get Width and Height of the Objects by applying the Ratio pixel to cm
            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio

            # Display rectangle
            bx = cv2.boxPoints(rect)
            bx = np.int0(bx)

            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.polylines(frame, [bx], True, (255, 0, 0), 2)
            cv2.putText(frame, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            cv2.putText(frame, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

    cv2.imshow("Camera", frame)

    # Quit(press q)
    if cv2.waitKey(1) & 0XFF == ord("q"):                        
        break

cam.release()
cv2.destroyAllWindows()
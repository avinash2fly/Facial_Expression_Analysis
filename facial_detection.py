import cv2
import numpy as np

# path = 'haarcascade/'

# face_cascade = cv2.CascadeClassifier(path+'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(path+'haarcascade_eye.xml')

# cap = cv2.VideoCapture(0);

# while True:
#     ret, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray,1.3,5)
#     print(faces)
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w, y+h), (255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for (ex,ey,ew,eh) in eyes:
#             cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh),(0,255,0), 2)

#     cv2.imshow('img', img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()


# ------------------------------------------------------------------------------------ #

import dlib
import tensorflow as tf
from imutils import face_utils
from imutils.face_utils import rect_to_bb


# ------------------------------------------------------------------------------------- #
# GLOBAL VARIABLES
# ------------------------------------------------------------------------------------- #

face_pred_path = "shape_predictor_68_face_landmarks.dat"

LEFT_EYE_IDXS = [i for i in range(36, 42)]
RIGHT_EYE_IDXS = [i for i in range(42, 48)]

target_width = 48
target_height = target_width

zoom = 0.2

target_left_eye_pos = [zoom, 0.35]   # [x,y] coordinate
target_right_eye_pos = [1-zoom, 0.35]
target_distance = target_width*(target_right_eye_pos[0] - target_left_eye_pos[0])

img_shape = (target_width,target_height)

# ------------------------------------------------------------------------------------- #
# FUNCTIONS
# ------------------------------------------------------------------------------------- #


def detect_facial_landmarks(img, detector, draw=False, drawIdx = None):
    landmarks = []
    faces = []

    # Detect face region using dlib
    rects = detector(img, 0)
     
    # loop over each face detected
    for (i, rect) in enumerate(rects):

        # Detect facial landmarks in the given face region
        points = predictor(img, rect)
        points = face_utils.shape_to_np(points)

        faces.append(rect)
        landmarks.append(points)

        # Draw circle for each facial landmarks detected
        if draw:
            if drawIdx is not None:
                drawPoints = points[drawIdx]
            else:
                drawPoints = points
            for (x, y) in drawPoints:
                cv2.circle(img, (x, y), 2, (255, 255, 0), -1)

    return faces, landmarks



# ------------------------------------------------------------------------------------- #
# MAIN
# ------------------------------------------------------------------------------------- #


p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0);

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # ------------------------------------------------------------------------------------- #
    #   *****************************    IMAGE PROCESSING    *****************************  #
    # ------------------------------------------------------------------------------------- #

    # ---- 1. Histogram equalization ------------------------------------------------------ #

    img = cv2.equalizeHist(gray)

    if img.shape[1] < target_width:
        target_width = img.shape[1]
    if img.shape[0] < target_height:
        target_height = img.shape[0]

    # 2.1 Detect face landmarks using dlib
    faces, landmarks = detect_facial_landmarks(img, detector, draw=False, drawIdx=None )#LEFT_EYE_IDXS+RIGHT_EYE_IDXS)

    for face, landmark in zip(faces, landmarks):

        # 2.2 Calculate the centre of left and right eyes
        left_eye = landmark[LEFT_EYE_IDXS,]
        right_eye = landmark[RIGHT_EYE_IDXS,]
        left_centre = np.mean(left_eye,axis=0)
        right_centre = np.mean(right_eye,axis=0)
        slope = right_centre - left_centre

        # 2.3 Get scaling factor by taking the ratio of the target distance between the centre of
        #     the left and right eyes with the actual distnace between the left and right eyes. 
        distance = np.sqrt((slope[0] ** 2) + (slope[1] ** 2))
        scale = target_distance / distance

        # 2.4 Calculate the angle 
        angle = np.degrees(np.arctan2(slope[1], slope[0]))

        # 2.5 Get rotation matrix based on the calculated scaling factor and angle
        pivot = (left_centre+right_centre)//2
        # M = cv2.getRotationMatrix2D(tuple(pivot.astype(int)), angle, scale)

        # 2.6 Update transalation part
        # tX = target_width * 0.5
        # tY = target_height * target_left_eye_pos[1]
        # M[0, 2] += (tX - pivot[0])
        # M[1, 2] += (tY - pivot[1])

        # Draw centre of left eye
        cv2.circle(img, tuple(left_centre.astype(int)), 2, (255, 255, 0), -1)
        # # Draw centre of right eye
        cv2.circle(img, tuple(right_centre.astype(int)), 2, (255, 255, 0), -1)
        # # Draw centre of rotation
        cv2.circle(img, tuple(pivot.astype(int)), 2, (255, 255, 0), -1)
        # # Draw a line between eyes
        cv2.line(img,tuple(left_centre.astype(int)),tuple(right_centre.astype(int)),(255,255,0),2)
        # # Draw square
        cv2.rectangle(img,(face.left(),face.bottom()),(face.right(),face.top()),(255,255,0),2)

        # apply the affine transformation
        # final_img = cv2.warpAffine(img, M, (target_width, target_height), flags=cv2.INTER_CUBIC)


    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
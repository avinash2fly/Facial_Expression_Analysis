
# Import libraries

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd 

from imutils import face_utils
from imutils.face_utils import rect_to_bb



# ------------------------------------------------------------------------------------- #
# GLOBAL VARIABLES
# ------------------------------------------------------------------------------------- #
output_file = './Data/CK_plus.csv'
img_Path = "./Data/CK+/cohn-kanade-images"
label_Path = "./Data/CK+/Emotion"
face_pred_path = "shape_predictor_68_face_landmarks.dat"

LEFT_EYE_IDXS = [i for i in range(36, 42)]
RIGHT_EYE_IDXS = [i for i in range(42, 48)]

target_width = 48
target_height = target_width

target_left_eye_pos = [0.35, 0.35]   # [x,y] coordinate
target_right_eye_pos = [0.65, 0.35]
target_distance = target_width*(target_right_eye_pos[0] - target_left_eye_pos[0])

img_shape = (target_width,target_height)
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



def align_face(img):
	pass

# ------------------------------------------------------------------------------------- #

# Define dlib facial landmarks detector and predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


output = open(output_file, 'w')
output.write('pixels,emotion,Usage\n')


# Directory of all the images
object_Dirs = os.listdir(img_Path)
try:
	object_Dirs.remove('.DS_Store')
except:
	pass

# Directory of all the labels
label_Dirs = os.listdir(label_Path)
try:
	label_Dirs.remove('.DS_Store')
except:
	pass


# Go through each object's (people) folder
for object_img_path, object_label_path in zip(object_Dirs, label_Dirs):

	# Each object has a few folders corresponding to each emotions (not all objects have all emotions)
	object_emo_dirs = os.listdir(os.path.join(img_Path, object_img_path))
	try:
		object_emo_dirs.remove('.DS_Store')
	except:
		pass

	objects_labels_dirs = os.listdir(os.path.join(label_Path, object_label_path))
	try:
		objects_labels_dirs.remove('.DS_Store')
	except:
		pass

	# Go through each emotion folder under each object
	for emo_img_path, emo_label_path in zip(object_emo_dirs, objects_labels_dirs):

		emo_img_files = os.listdir(os.path.join(img_Path, object_img_path, emo_img_path))
		emo_label_files = os.listdir(os.path.join(label_Path, object_label_path, emo_label_path))

		emo_img_files.sort(reverse=True)

		if len(emo_img_files)>0 and len(emo_label_files)>0:

			#  Read image and label 
			img = cv2.imread(os.path.join(img_Path, object_img_path, emo_img_path, emo_img_files[0]),0)
			label = open(os.path.join(label_Path, object_label_path, emo_label_path, emo_label_files[0]), 'r').read().strip()[0]

			# ------------------------------------------------------------------------------------- #
			#   *****************************    IMAGE PROCESSING    *****************************  #
			# ------------------------------------------------------------------------------------- #

			# ---- 1. Histogram equalization ------------------------------------------------------ #

			img = cv2.equalizeHist(img)


			# ---- 2. Facial alignment ------------------------------------------------------------ #

			if img.shape[1] < target_width:
				target_width = img.shape[1]
			if img.shape[0] < target_height:
				target_height = img.shape[0]

			# 2.1 Detect face landmarks using dlib
			faces, landmarks = detect_facial_landmarks(img, detector, draw=False, drawIdx=LEFT_EYE_IDXS+RIGHT_EYE_IDXS)

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
				M = cv2.getRotationMatrix2D(tuple(pivot.astype(int)), angle, scale)

				# 2.6 Update transalation part
				tX = target_width * 0.5
				tY = target_height * target_left_eye_pos[1]
				M[0, 2] += (tX - pivot[0])
				M[1, 2] += (tY - pivot[1])

				# Draw centre of left eye
				# cv2.circle(img, tuple(left_centre.astype(int)), 2, (255, 255, 0), -1)
				# # Draw centre of right eye
				# cv2.circle(img, tuple(right_centre.astype(int)), 2, (255, 255, 0), -1)
				# # Draw centre of rotation
				# cv2.circle(img, tuple(pivot.astype(int)), 2, (255, 255, 0), -1)
				# # Draw a line between eyes
				# cv2.line(img,tuple(left_centre.astype(int)),tuple(right_centre.astype(int)),(255,255,0),2)
				# # Draw square
				# cv2.rectangle(img,(face.left(),face.bottom()),(face.right(),face.top()),(255,255,0),2)

				# apply the affine transformation
				final_img = cv2.warpAffine(img, M, (target_width, target_height), flags=cv2.INTER_CUBIC)

				# (x, y, w, h) = rect_to_bb(face)
				# faceOrig = cv2.resize(img[y:y + h, x:x + w], img_shape)

				# fig = plt.figure()
				# fig.subplots_adjust(hspace=0.3)

				# plt.subplot(1, 2, 1)
				# plt.imshow(faceOrig)
				# plt.subplot(1, 2, 2)
				# plt.imshow(output)

				# plt.show()

				# ------------------------------------------------------------------------------------- #
				# 3. Write results to csv

				pixels = ' '.join(final_img.flatten().astype('str'))
				label = int(label)-1

				output.write(pixels+','+str(label)+',Training\n')

output.close()


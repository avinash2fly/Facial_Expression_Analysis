from __future__ import print_function
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB

import cv2
import numpy as np
import os
import pandas as pd
import time

import matplotlib.pyplot as plt

# ************************** CONSTANTS ************************** #
N_CLUSTER = 4096
IMG_SHAPE = [48, 48]




# ************************** FUNCTIONS ************************** #

def extractKeyPoints(data):
	'''
	data : a list of grayscale images as np.array object

	It returns a list of keypoints, descriptors]
	'''

	# Array where we will store [keypoints, descriptors] list.
	key_features = []

	# Initialise SIFT detector
	detector = cv2.xfeatures2d.SIFT_create()

	# Obtain key points and descriptors for each image
	for img in data:
		if len(img.shape) == 1:
			img = img.reshape(tuple(IMG_SHAPE))
		keypoints, descriptors = detector.detectAndCompute(img, None)
		key_features.append([keypoints, descriptors])

	return key_features




# ************************** MAIN CODE ************************** #

if __name__ == "__main__":
	
	# -------------------------- Read Images ------------------------ #

	# filepath = os.path.abspath(os.path.join('./Data/fer2013/', 'fer2013.csv'))
	filepath = os.path.abspath(os.path.join('./', 'fer20131000.csv'))

	dt = pd.read_csv(filepath, sep=',', header=0)

	train_dt = dt.loc[dt['Usage'] == 'Training', :]
	# train_dt = train_dt[:256]
	# validation_dt = dt.loc[dt['Usage'] == 'PrivateTest', :]
	# test_dt = dt.loc[dt['Usage'] == 'PublicTest', :]

	# Convert labels into 1 hot encoding
	labels = np.zeros((train_dt['emotion'].shape[0], 7))
	labels[np.arange(labels.shape[0]), train_dt['emotion']] = 1



	# Data need to be uint8 type or else SIFT/SURF does not work
	data = [np.array(dtpoint.split(), dtype= np.uint8) for dtpoint in train_dt['pixels']]




	# ------------------- Bag of features processing ---------------- #

	# Step 1 : Extract key points and descriptors
	data_kp = extractKeyPoints(data)

	# Step 2 : Combine all descriptors into 1 array
	descriptors = None
	for img in data_kp:
		for desc in img[1]:
			if descriptors is None:
				descriptors = desc
				continue
			descriptors = np.vstack((descriptors, desc))

	# Step 3 : Cluster descriptors using KMeans
	clusterer = KMeans(n_clusters=N_CLUSTER, random_state=0)
	clusterer.fit(descriptors)

	# Step 4 : Relabel and create histogram
	SIFT_data = [[],train_dt['emotion']]
	bins = [i for i in range(N_CLUSTER)]
	anchor = 0

	for img in data_kp:
		n_kp = len(img[0])
		hist = np.histogram(clusterer.labels_[anchor:anchor+n_kp], bins=bins)
		SIFT_data[0].append(hist[0])
		anchor += n_kp

	SIFT_data[0] = np.array(SIFT_data[0])


# scores = {}
# for N_CLUSTER in range(10,1020,20):
# 	# Step 3 : Cluster descriptors using KMeans
# 	clusterer = KMeans(n_clusters=N_CLUSTER, random_state=0)
# 	clusterer.fit(descriptors)

# 	# Step 4 : Relabel and create histogram
# 	SIFT_data = [[],train_dt['emotion']]
# 	bins = [i for i in range(N_CLUSTER)]
# 	anchor = 0

# 	for img in data_kp:
# 		n_kp = len(img[0])
# 		hist = np.histogram(clusterer.labels_[anchor:anchor+n_kp], bins=bins)
# 		SIFT_data[0].append(hist[0])
# 		anchor += n_kp

# 	SIFT_data[0] = np.array(SIFT_data[0])

# 	predictor = GaussianNB()
# 	predictor.fit(SIFT_data[0], SIFT_data[1])

# 	scores[N_CLUSTER] = predictor.score(SIFT_data[0], SIFT_data[1])

# 	print('N_CLUSTER: {}, Score: {}'.format(N_CLUSTER, scores[N_CLUSTER]))







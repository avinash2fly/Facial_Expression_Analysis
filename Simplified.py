from __future__ import print_function, division

from datetime import datetime

from imutils import face_utils
from imutils.face_utils import rect_to_bb

from math import sqrt

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time

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

img_shape = [target_width,target_height]
N_CLUSTER = 4097



# ------------------------------------------------------------------------------------- #
# FUNCTIONS
# ------------------------------------------------------------------------------------- #




def accuracy(sess, data, batches, batch_size, X, Y, accuracy_op):
    # compute number of batches for given batch_size
    n_batches = len(batches)

    overall_accuracy = 0.0
    for i in range(n_batches):
        batch = batches[i]
        accuracy_batch = \
            sess.run(accuracy_op, feed_dict={X: batch[0], Y: batch[1], SIFT: batch[2]})
        overall_accuracy += accuracy_batch
    # print(overall_accuracy)
    return overall_accuracy / n_batches



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


class InputError(Exception):
	"""Exception raised for errors in the input.
    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

	def __init__(self, message):
		super().__init__(message)


def get_accuracy_op(preds_op, Y):
    with tf.name_scope('accuracy_ops'):
        correct_preds_op = tf.equal(tf.argmax(preds_op, 1), tf.argmax(Y, 1))
        # the tf.cast sets True to 1.0, and False to 0.0. With N predictions, of
        # which M are correct, the mean will be M/N, i.e. the accuracy
        accuracy_op = tf.reduce_mean(tf.cast(correct_preds_op, tf.float32))
    return accuracy_op


def get_batch(data, batch_size):
	'''
	Function which returns a list of batch, each with batch_size data points.
	'''

	indices = np.array(range(0, data[0].shape[0]))
	np.random.shuffle(indices)

	split = list(range(0, data[0].shape[0], batch_size))[1:]
	if data[0].shape[0] - split[-1] < batch_size:
		split.pop()
	split_indices = np.split(indices, split)

	batches = []

	for idx in split_indices:
		batches.append([data[0][idx, :], data[1][idx, :], data[2][idx, :]])

	return batches


def getCluster(data_kp, n_cluster = N_CLUSTER):
	'''
	Return fitted clusterer.
	'''
	# ------------------- Bag of features processing ---------------- #

	# print('data min: {}, max: {}'.format(np.min(data), np.max(data)))

	# Step 2 : Combine all descriptors into 1 array
	descriptors = None
	for img in data_kp:
		if len(img[0]) == 0:
			continue
		for desc in img[1]:
			if descriptors is None:
				descriptors = desc
				continue
			descriptors = np.vstack((descriptors, desc))

	if descriptors is None:
		return None, None

	# Step 3 : Cluster descriptors using KMeans
	clusterer = KMeans(n_clusters=n_cluster, random_state=0)
	clusterer.fit(descriptors)

	return clusterer


def convert2bagOfFeatures(data_kp, clusterer, n_cluster = N_CLUSTER):
	'''
	Function to relabel feature.
	data_kp is a list of set of keypoints and descriptors of each image.
	data_kp = [ [keypoints_img1, descriptors_img1], [keypoints_img2, descriptors_img2], ...  ]
	keypoints_img1 = [kp1, kp2, .... ]
	descriptors_img1 = [desc1, desc2, ...]
	'''
	SIFT_data = []
	bins = [i for i in range(n_cluster)]
	anchor = 0

	for img in data_kp:
		n_kp = len(img[0])
		hist = np.histogram(clusterer.labels_[anchor:anchor+n_kp], bins=bins)
		SIFT_data.append(hist[0])
		anchor += n_kp

	SIFT_data = np.array(SIFT_data)

	return SIFT_data



def getSiftFeatures(data, n_cluster = N_CLUSTER):
	# ------------------- Bag of features processing ---------------- #

	# Step 1 : Extract key points and descriptors
	data_kp = extractKeyPoints(data)
	print('data min: {}, max: {}'.format(np.min(data), np.max(data)))

	# Step 2 : Combine all descriptors into 1 array
	descriptors = None
	for img in data_kp:
		for desc in img[1]:
			if descriptors is None:
				descriptors = desc
				continue
			descriptors = np.vstack((descriptors, desc))

	# if n_cluster > len(data):
	# 	n_cluster = int(sqrt(len(data)))

	# Step 3 : Cluster descriptors using KMeans
	clusterer = KMeans(n_clusters=n_cluster, random_state=0)
	clusterer.fit(descriptors)

	# Step 4 : Relabel and create histogram
	SIFT_data = []
	bins = [i for i in range(n_cluster)]
	anchor = 0

	for img in data_kp:
		n_kp = len(img[0])
		hist = np.histogram(clusterer.labels_[anchor:anchor+n_kp], bins=bins)
		SIFT_data.append(hist[0])
		anchor += n_kp

	SIFT_data = np.array(SIFT_data)

	return SIFT_data


def extractKeyPoints(data, shape=tuple(img_shape)):
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
			img = img.reshape(shape)
		# print(img.shape)
		keypoints, descriptors = detector.detectAndCompute(img, None)
		key_features.append([keypoints, descriptors])

	return key_features


# ------------------------------------------------------------------------------------- #
# NEURAL NET STRUCTURE
# ------------------------------------------------------------------------------------- #

def NeuralNet(X, Y, SIFT, outputsize=10, padding="same", SIFT_size = 2048):

	num_features = 32 # The number of output nodes
	kernel = 5        # The size of the convolution kernel

	# Initializer for weights and bias
	kernel_initializer = tf.truncated_normal_initializer()
	bias_initializer = tf.zeros_initializer()

	activation = tf.nn.leaky_relu   # Activation function for each layer
	regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)  # Regularizer


	# ------------------------ Convolutional layer 1 ----------------------------

	layer1_conv1 = tf.layers.conv2d(X,  filters=num_features, 
										kernel_size=kernel, 
										kernel_initializer=kernel_initializer,
										use_bias=True, 
										bias_initializer=bias_initializer,
										padding=padding, 
										activation=activation)

	layer1_norm1 = tf.layers.batch_normalization(layer1_conv1)

	layer1_conv2 = tf.layers.conv2d(layer1_norm1, filters=num_features, 
										kernel_size=kernel, 
										kernel_initializer=kernel_initializer,
										use_bias=True, 
										bias_initializer=bias_initializer,
										padding=padding, 
										activation=activation)

	layer1_norm2 = tf.layers.batch_normalization(layer1_conv2)

	layer1_maxPool = tf.layers.max_pooling2d(layer1_norm2,pool_size=(2,2),strides=(2,2))
	layer1_dropout = tf.layers.dropout(layer1_maxPool,rate=0.5)


	# ------------------------ Convolutional layer 2 ----------------------------
	num_features *= 2
	layer2_conv1 = tf.layers.conv2d(layer1_dropout, filters=num_features,
	                                kernel_size=kernel,
	                                padding=padding,
	                                activation=activation, 
	                                use_bias=True,
	                                bias_initializer=bias_initializer,
	                                kernel_initializer=kernel_initializer
	                               )
	layer2_norm1 = tf.layers.batch_normalization(layer2_conv1)
	layer2_conv2 = tf.layers.conv2d(layer2_norm1, filters=num_features,
	                                kernel_size=kernel,
	                                padding=padding,
	                                activation=activation, 
	                                use_bias=True,
	                                bias_initializer=bias_initializer,
	                                kernel_initializer=kernel_initializer
	                               )
	layer2_norm2 = tf.layers.batch_normalization(layer2_conv2)
	layer2_maxPool = tf.layers.max_pooling2d(layer2_norm2, pool_size=(2, 2), strides=(2, 2))
	layer2_dropout = tf.layers.dropout(layer2_maxPool, rate=0.5)


	# ------------------------ Convolutional layer 3 ----------------------------
	num_features *= 2
	layer3_conv1 = tf.layers.conv2d(layer2_dropout, filters=num_features,
	                                kernel_size=kernel,
	                                padding=padding,
	                                activation=activation, 
	                                use_bias=True,
	                                bias_initializer=bias_initializer,
	                                kernel_initializer=kernel_initializer
	                                )
	layer3_norm1 = tf.layers.batch_normalization(layer3_conv1)
	layer3_conv2 = tf.layers.conv2d(layer3_norm1, filters=num_features,
	                                kernel_size=kernel,
	                                padding=padding,
	                                activation=activation, 
	                                use_bias=True,
	                                bias_initializer=bias_initializer,
	                                kernel_initializer=kernel_initializer
	                                )
	layer3_norm2 = tf.layers.batch_normalization(layer3_conv2)
	layer3_maxPool = tf.layers.max_pooling2d(layer3_norm2, pool_size=(2, 2), strides=(2, 2))
	last_layer = tf.layers.dropout(layer3_maxPool, rate=0.5)


	# ------------------------- Fully connected layer ----------------------------

	dim = int(last_layer.get_shape()[1] * last_layer.get_shape()[2] * last_layer.get_shape()[3])

	flatten_layer = tf.reshape(last_layer, [tf.shape(last_layer)[0], dim])

	num_features = SIFT_size

	dense3 = tf.layers.dense(flatten_layer, num_features, activation=activation)
	dense3_drop = tf.layers.dropout(dense3, rate=0.4)

	# ------------------------- SIFT Layer 1 ------------------------------

	# denseSIFT1 = tf.layers.dense(SIFT, num_features, kernel_initializer=kernel_initializer, 
	# 							 use_bias=True, bias_initializer=bias_initializer, activation=activation)
	# denseSIFT1_drop = tf.layers.dropout(denseSIFT1, rate=0.5)


	# ------------------------- Combined Layer ------------------------------
	# dense_combined = tf.reduce_mean([denseSIFT1_drop, dense3_drop], 0)  # [1.5, 1.5]


	logits = tf.layers.dense(inputs=dense3_drop,units=outputsize)

	# preds will hold the predicted class
	preds = tf.nn.softmax(logits)

	# Cross entropy
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
	total_loss = tf.reduce_mean(cross_entropy)

	return logits, preds, total_loss


# ------------------------------------------------------------------------------------- #
# DATA PREPROCESSING
# ------------------------------------------------------------------------------------- #

def preprocess(data):
	'''
	Assuming data = [data_points, labels]
	'''
	data_new = []

	for img in data:
		# img_new = cv2.equalizeHist(img)
		data_new.append(img/255.0)

	return np.array(data_new)


# def equalize(data):
# 	data_new = []

# 	for img in data:
# 		data_new.append(cv2.equalizeHist(img))

# 	return np.array(data_new)


# ------------------------------------------------------------------------------------- #
# TRAINING
# ------------------------------------------------------------------------------------- #

def trainNN(data, N_CLUSTER=N_CLUSTER, tensorboard_name="9517 Test"):

	input_dim = data[0].shape[1]
	output_dim = data[1].shape[1]

	n_epochs = 100
	batch_size = 64
	learning_rate = 0.005

	if N_CLUSTER > data[0].shape[0]:
		N_CLUSTER = int(sqrt(data[0].shape[0]))

	# Define dimension of input, X and output Y
	X = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name="image_input")
	Y = tf.placeholder(dtype=tf.float32, shape=[None, output_dim], name="image_target_onehot")
	SIFT = tf.placeholder(dtype=tf.float32, shape=[None, N_CLUSTER-1], name="SIFT_input")

	# Define the CNN
	logits_op, preds_op, loss_op = NeuralNet(tf.reshape(X, [-1, img_shape[0], img_shape[1], 1]), Y, SIFT, outputsize=output_dim, SIFT_size=2048)
	tf.summary.histogram('pre_activations', logits_op)

	optimizer = tf.train.AdamOptimizer  # ADAM - widely used optimiser (ref: http://arxiv.org/abs/1412.6980)
	train_op = optimizer(learning_rate).minimize(loss_op)

	# Prediction and accuracy ops
	accuracy_op = get_accuracy_op(preds_op, Y)

	# TensorBoard for visualisation
	# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
	summaries_op = tf.summary.merge_all()

	# Separate accuracy summary so we can use train and test sets
	accuracy_placeholder = tf.placeholder(shape=[], dtype=tf.float32)
	accuracy_summary_op = tf.summary.scalar("accuracy", accuracy_placeholder)

	# When run, the init_op initialises any tensorflow variables
	init_op = tf.global_variables_initializer()

	# Start session
	sess = tf.Session()
	sess.run(init_op)


	# Initialise TensorBoard Summary writers
	dtstr = "{:%b_%d_%H-%M-%S}".format(datetime.now())
	train_writer = tf.summary.FileWriter('./summaries/' + tensorboard_name + '_' + dtstr + '/train', sess.graph)
	test_writer = tf.summary.FileWriter('./summaries/' + tensorboard_name + '_' + dtstr + '/test')

	print('Starting Training...')

	# record starting time
	train_start = time.time()
	saver = tf.train.Saver()

	X_train, X_test, y_train, y_test, SIFT_train, SIFT_test = train_test_split(data[0], data[1], data[2], test_size=0.30)

	train_data = [X_train, y_train]
	test_data = [X_test, y_test]


	# Step 1 : Extract key points and descriptors
	data_kp_train = extractKeyPoints(SIFT_train)
	data_kp_test = extractKeyPoints(SIFT_test)

	# Get descriptors and get clusters
	clusterer = getCluster(data_kp_train, n_cluster=N_CLUSTER)

	# Get bag of features count for each image
	SIFT_data_train = convert2bagOfFeatures(data_kp_train, clusterer, n_cluster = N_CLUSTER)
	SIFT_data_test = convert2bagOfFeatures(data_kp_test, clusterer, n_cluster = N_CLUSTER)

	# Append result to train_data
	train_data.append(SIFT_data_train/N_CLUSTER)
	test_data.append(SIFT_data_test/N_CLUSTER)




	# Run through the entire dataset n_training_epochs times
	train_loss=100;
	for i in range(n_epochs):
		# Initialise statistics
		training_loss = 0
		epoch_start = time.time()

		batches = get_batch(train_data, batch_size)
		t_batches = get_batch(test_data, 10)
		n_batches = len(batches)

		#

		# Run the SGD train op for each minibatch
		for j in range(n_batches):
			batch = batches[j]

			# Run a training step
			trainstep_result, batch_loss, summary = \
			    sess.run([train_op, loss_op, summaries_op], feed_dict={X: batch[0], Y: batch[1], SIFT: batch[2]})
			train_writer.add_summary(summary, j)
			training_loss += batch_loss

		# Timing and statistics
		epoch_duration = round(time.time() - epoch_start, 2)
		ave_train_loss = training_loss / n_batches

		# Get accuracy
		train_accuracy = \
		    accuracy(sess, train_data, batches, batch_size, X, Y, accuracy_op)
		test_accuracy = \
		    accuracy(sess, test_data, t_batches, batch_size, X, Y, accuracy_op)

		if train_loss > ave_train_loss:
			save_path = saver.save(sess, "./models/model.ckpt")
			train_loss = ave_train_loss
			print("saved checkpoint")

		# log accuracy at the current epoch on training and test sets
		train_acc_summary = sess.run(accuracy_summary_op,
		                             feed_dict={accuracy_placeholder: train_accuracy})
		train_writer.add_summary(train_acc_summary, i)
		test_acc_summary = sess.run(accuracy_summary_op,
		                            feed_dict={accuracy_placeholder: test_accuracy})
		test_writer.add_summary(test_acc_summary, i)
		[writer.flush() for writer in [train_writer, test_writer]]

		train_duration = round(time.time() - train_start, 2)
		# Output to montior training
		print('Epoch {0}, Training Loss: {1:.5f}, Test accuracy: {2:.5f}, \
	time: {3}s, total time: {4}s'.format(i, ave_train_loss,
	                                 test_accuracy, epoch_duration,
	                                 train_duration))
	print('Total training time: {0}s'.format(train_duration))
	print('Confusion Matrix:')
	true_class = tf.argmax(Y, 1)
	predicted_class = tf.argmax(preds_op, 1)
	cm = tf.confusion_matrix(predicted_class, true_class)
	print(sess.run(cm, feed_dict={X: test_data[0],
	                              Y: test_data[1],
	                              SIFT: test_data[2]}))




	print('Training Complete')
	sess.close()





if __name__ == "__main__":
	filepath = os.path.abspath(os.path.join('./Data/', 'CK_plus.csv'))

	raw_data = pd.read_csv(filepath, sep=',', header=0)

	raw_data_pixels = raw_data.loc[raw_data['Usage'] == 'Training', ['pixels']]

	pixels = []

	for dt in raw_data_pixels['pixels']:
		dt = np.array([int(pixel) for pixel in dt.split(' ')], dtype=np.uint8)
		eq_dt = cv2.equalizeHist(dt)
		eq_dt = eq_dt.reshape(eq_dt.shape[0])
		pixels.append(eq_dt)

	labels = np.zeros((raw_data_pixels.shape[0], 7), dtype=np.float)

	labels[np.arange(labels.shape[0]), raw_data.loc[raw_data['Usage'] == 'Training', ['emotion']]] = 1.0

	data = [np.array(pixels, dtype=np.float)/255.0, labels, np.array(pixels)]

	tensorboard_name = "Simplified"

	trainNN(data, tensorboard_name=tensorboard_name)
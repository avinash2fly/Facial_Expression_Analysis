from datetime import datetime

from math import sqrt

from imutils import face_utils
from imutils.face_utils import rect_to_bb

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import cv2
import dlib
import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import time

face_pred_path = "shape_predictor_68_face_landmarks.dat"

LEFT_EYE_IDXS = [i for i in range(36, 42)]
RIGHT_EYE_IDXS = [i for i in range(42, 48)]

target_width = 48
target_height = target_width

zoom = 0

target_left_eye_pos = [zoom, 0.35]   # [x,y] coordinate
target_right_eye_pos = [1-zoom, 0.35]
target_distance = target_width*(target_right_eye_pos[0] - target_left_eye_pos[0])

img_shape = (target_width,target_height)
N_CLUSTER = 18


n_training_epochs = 100
batch_size = 64
learning_rate = 0.005



class InputError(Exception):
    """Exception raised for errors in the input.
    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        super().__init__(message)




# ************************** FUNCTIONS ************************** #

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


def Block(input_layer, num_features, activation, bias_initializer, kernel_initializer, regularizer, dropout=0.5, residual=False, kernel=3, padding='same'):
    '''
    Set residual=True for residual block
    '''
    conv1 = tf.layers.conv2d(input_layer, filters=num_features,
                             kernel_size=kernel,
                             padding=padding,
                             activation=activation, 
                             use_bias=True,
                             bias_initializer=bias_initializer,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=regularizer)
    conv1 = tf.layers.batch_normalization(conv1)


    conv2 = tf.layers.conv2d(conv1, filters=num_features,
                             kernel_size=kernel,
                             padding=padding,
                             activation=activation, 
                             use_bias=True,
                             bias_initializer=bias_initializer,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=regularizer)
    conv2 = tf.layers.batch_normalization(conv2)

    if residual:
        identity = tf.layers.conv2d(input_layer, filters=num_features, kernel_size=1, padding=padding)
        identity = tf.layers.batch_normalization(identity)

        conv2 = tf.add(identity, conv2)
        conv2 = tf.nn.leaky_relu(conv2)

    maxPool = tf.layers.max_pooling2d(conv2,pool_size=(2,2),strides=(2,2))
    dropout = tf.layers.dropout(maxPool,rate=dropout)

    return dropout


def NeuralNet(X, Y, SIFT, convlayer_sizes=[10, 10], filter_shape=[3, 3], outputsize=10, padding="same", SIFT_size = 2048):
    """
    logits: The inputs to the activation function
    preds: The outputs of the activation function (a probability
    distribution over the 10 digits)
    batch_xentropy: The cross-entropy loss for each image in the batch
    batch_loss: The average cross-entropy loss of the batch
    """

    num_features = 32
    kernel = 3
    kernel_initializer = tf.truncated_normal_initializer()
    # bias_initializer = tf.truncated_normal_initializer()
    # kernel_initializer = None
    bias_initializer = None
    # activation = tf.nn.leaky_relu
    activation = tf.nn.relu
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

    layer1 = Block(X, num_features, activation, bias_initializer, kernel_initializer, regularizer,
        dropout=0.5, residual=False, kernel=3, padding=padding)

    # # --------------------------layer 2-----------------------------------
    num_features *= 2

    layer2 = Block(layer1, num_features, activation, bias_initializer, kernel_initializer, regularizer,
        dropout=0.5, residual=False, kernel=3, padding=padding)

    # --------------------------layer 3----------------------------------
    num_features *= 2

    layer3 = Block(layer2, num_features, activation, bias_initializer, kernel_initializer, regularizer,
        dropout=0.5, residual=False, kernel=3, padding=padding)


    dim = int(layer3.get_shape()[1] * layer3.get_shape()[2] * layer3.get_shape()[3])


    # -------------------------Fully connected layer ----------------------
    dense = tf.reshape(layer3, [tf.shape(layer3)[0], dim])

    num_features = SIFT_size
    # dense1 = tf.layers.dense(X,num_features,activation=tf.nn.leaky_relu)
    # dense1_drop = tf.layers.dropout(dense1, rate=0.5)

    # num_features //= 2
    # dense2 = tf.layers.dense(dense1_drop, num_features, activation=tf.nn.leaky_relu)
    # dense2_drop = tf.layers.dropout(dense2, rate=0.5)

    # num_features //= 2
    dense = tf.layers.dense(dense, num_features, activation=activation)
    dense = tf.layers.dropout(dense, rate=0.4)


    # ------------------------- SIFT Layer 1 ------------------------------

    denseSIFT = tf.layers.dense(SIFT,num_features,
                                activation=activation, 
                                kernel_regularizer=regularizer
                                )
    denseSIFT_drop = tf.layers.dropout(denseSIFT, rate=0.5)


    # ------------------------- Combined Layer ------------------------------
    dense_combined = tf.reduce_mean([denseSIFT_drop, dense], 0)  # [1.5, 1.5]


    # Pre-activation
    logits = tf.layers.dense(inputs=dense_combined,units=outputsize)

    # Pass logits through activation function
    preds = tf.nn.softmax(logits)

    # Cross entropy
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    batch_loss = tf.reduce_mean(batch_xentropy)

    return logits, preds, batch_loss

    # import models_py.stanford_2 as md

    # return md.NeuralNet(X, Y, SIFT, convlayer_sizes=convlayer_sizes, filter_shape=filter_shape, outputsize=outputsize, padding=padding, SIFT_size = SIFT_size)

    



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



def detect_facial_landmarks(img, detector, predictor, draw=False, drawIdx = None):
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
        if descriptors is None:
            descriptors = img[1]
            continue
        descriptors = np.vstack((descriptors, img[1]))

    if descriptors is None:
        return None, None

    # Step 3 : Cluster descriptors using KMeans
    clusterer = KMeans(n_clusters=n_cluster, random_state=0)
    clusterer.fit(descriptors)

    return clusterer



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
    #   n_cluster = int(sqrt(len(data)))

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



def preprocess(image, detector, predictor, zoom=0, target_width=None, LEFT_EYE_IDXS=LEFT_EYE_IDXS, RIGHT_EYE_IDXS=RIGHT_EYE_IDXS):
    '''
    This function does:
    1. Histogram equalization
    2. Face alignment
    3. Crop face into image of size (target_width,target_width)
    '''

    target_left_eye_pos = [zoom, 0.35]   # [x,y] coordinate
    target_right_eye_pos = [1-zoom, 0.35]

    target_height = target_width

    if target_width is None:
        target_width = image.shape[1]
        target_height = image.shape[0]

    target_distance = target_width*(target_right_eye_pos[0] - target_left_eye_pos[0])
    
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    processed_images = []

    # 1 Histogram equalization

    img = cv2.equalizeHist(image)

    # 2 Detect face landmarks using dlib
    faces, landmarks = detect_facial_landmarks(img, detector, predictor, draw=False, drawIdx=LEFT_EYE_IDXS+RIGHT_EYE_IDXS)

    for face, landmark in zip(faces, landmarks):
        # print('Face : ', face)

        # 3 Calculate the centre of left and right eyes
        left_eye = landmark[LEFT_EYE_IDXS,]
        right_eye = landmark[RIGHT_EYE_IDXS,]
        left_centre = np.mean(left_eye,axis=0)
        right_centre = np.mean(right_eye,axis=0)
        slope = right_centre - left_centre

        # 4 Get scaling factor by taking the ratio of the target distance between the centre of
        #     the left and right eyes with the actual distance between the left and right eyes. 
        distance = np.sqrt((slope[0] ** 2) + (slope[1] ** 2))
        scale = target_distance / distance
        shift_x = target_width
        shift_y = target_height
        if zoom == 0:
            scale = 1
            shift_x = img.shape[1]
            shift_y = img.shape[0]

        # 5 Calculate the angle 
        angle = np.degrees(np.arctan2(slope[1], slope[0]))

        # 6 Get rotation matrix based on the calculated scaling factor and angle
        pivot = (left_centre+right_centre)//2
        M = cv2.getRotationMatrix2D(tuple(pivot.astype(int)), angle, scale)

        # 7 Update transalation part of the rotation matrix
        tX = shift_x * 0.5
        tY = shift_y * target_left_eye_pos[1]
        M[0, 2] += (tX - pivot[0])
        M[1, 2] += (tY - pivot[1])

        # # Draw centre of left eye
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
        final_img = cv2.warpAffine(img, M, (target_width, target_width), flags=cv2.INTER_CUBIC)
        # print('Final image dimension ', final_img)

        # Detect face again on the now aligned image
        # rects = detector(final_img, 0)
        # print('Faces detected : ', len(rects))

        # for rect in rects:
        #     cropped_face = final_img[rect.top():rect.bottom(), rect.left():rect.right()]
        #     # print('target widht : ', target_width)
        #     cropped_face = cv2.resize(cropped_face, (target_width, target_width))
        #     processed_images.append(cropped_face)
        processed_images.append(final_img)

    return processed_images, faces



def plot(img, final_img):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.3)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(final_img)
    plt.show()
    


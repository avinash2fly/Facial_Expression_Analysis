from __future__ import print_function

from datetime import datetime

from math import sqrt

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import cv2
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import time

# import matplotlib.pyplot as plt

# ************************** CONSTANTS ************************** #
N_CLUSTER = 4097
img_shape = [48, 48]


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
    # w1 = tf.Variable(tf.truncated_normal([filter_shape[0], filter_shape[0]], stddev=1.0/math.sqrt(float(X.shape[1]))))


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


def train(sess, data, n_epochs, batch_size,
          summaries_op, accuracy_summary_op, train_writer, test_writer,
          X, Y, SIFT, train_op, loss_op, accuracy_op, ratio=[70, 20, 10], N=N_CLUSTER):

    # record starting time
    train_start = time.time()
    saver = tf.train.Saver()
    # train_data, test_data, validation_data = split_data(data, ratio)
    
    X_train, X_test, y_train, y_test, SIFT_train, SIFT_test = train_test_split(data[0], data[1], data[2], test_size=0.30)

    train_data = [X_train, y_train]
    test_data = [X_test, y_test]

    # if N > SIFT_train.shape[0]:
    #   N = int(sqrt(SIFT_train.shape[0]))

    # Step 1 : Extract key points and descriptors
    data_kp_train = extractKeyPoints(SIFT_train)
    data_kp_test = extractKeyPoints(SIFT_test)

    # Get descriptors and get clusters
    clusterer = getCluster(data_kp_train, n_cluster=N)
    pickle.dump( {'clusterer':clusterer, 'N_CLUSTER':N }, open( "./pickle_jar/pickle_objects.p", "wb" ) )
    
    # Get bag of features count for each image
    SIFT_data_train = convert2bagOfFeatures(data_kp_train, clusterer, n_cluster = N)
    SIFT_data_test = convert2bagOfFeatures(data_kp_test, clusterer, n_cluster = N)

    # Append result to train_data
    train_data.append(SIFT_data_train/N)
    test_data.append(SIFT_data_test/N)


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


# ************************** MAIN CODE ************************** #

# # Get the SIFT features 
# SIFT_data = getSiftFeatures(data)

if __name__ == "__main__":

    # -------------------------- Read Images ------------------------ #

    # filepath = os.path.abspath(os.path.join('./Data/', 'CK_plus_48.csv'))
    filepath = os.path.abspath(os.path.join('./Data/', 'CK_fer_data.csv'))

    train_dt = pd.read_csv(filepath, sep=',', header=0)

    # train_dt = dt.loc[dt['Usage'] == 'Training', :]

    output_dim = 3
    # Convert labels into 1 hot encoding
    labels = np.zeros((train_dt['emotion'].shape[0], output_dim))
    labels[np.arange(labels.shape[0]), train_dt['emotion']] = 1

    faces=[]
    raw_faces=[]

    for pixel_sequence in train_dt['pixels']:
        image = np.array([int(pixel) for pixel in pixel_sequence.split(' ')], dtype=np.uint8)
        raw_faces.append(image.copy())
        image = image.reshape(tuple(img_shape))

        image = cv2.equalizeHist(image)
        image = image / 255.0 # 4

        image = image.flatten()
        faces.append(image)

    data =[np.array(faces), labels, np.array(raw_faces)]
    print('data : {}, label : {}', data[0].shape, data[1].shape)

    input_dim = data[0].shape[1]
    # output_dim = data[1].shape[1]
    

    tensorboard_name = '3 Classes'


    n_training_epochs = 100
    batch_size = 64
    learning_rate = 0.005

    if N_CLUSTER > data[0].shape[0]:
        N_CLUSTER = int(sqrt(data[0].shape[0]))

    # Define dimension of input, X and output Y
    X = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name="image_input")
    Y = tf.placeholder(dtype=tf.float32, shape=[None, output_dim], name="image_target_onehot")
    # print('SIFT arg, N: ', N_CLUSTER)
    SIFT = tf.placeholder(dtype=tf.float32, shape=[None, N_CLUSTER-1], name="SIFT_input")

    # Define the CNN
    logits_op, preds_op, loss_op = NeuralNet(
        tf.reshape(X, [-1, img_shape[0], img_shape[1], 1]), Y, SIFT, convlayer_sizes=[output_dim, output_dim],
        outputsize=output_dim, SIFT_size=2048)
    tf.summary.histogram('pre_activations', logits_op)

    # The training op performs a step of stochastic gradient descent on a minibatch
    # optimizer = tf.train.GradientDescentOptimizer # vanilla SGD
    # optimizer = tf.train.MomentumOptimizer # SGD with momentum
    optimizer = tf.train.AdamOptimizer  # ADAM - widely used optimiser (ref: http://arxiv.org/abs/1412.6980)
    train_op = optimizer(epsilon=1.0).minimize(loss_op)

    # Prediction and accuracy ops
    accuracy_op = get_accuracy_op(preds_op, Y)

    # TensorBoard for visualisation
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    summaries_op = tf.summary.merge_all()

    # Separate accuracy summary so we can use train and test sets
    accuracy_placeholder = tf.placeholder(shape=[], dtype=tf.float32)
    accuracy_summary_op = tf.summary.scalar("accuracy", accuracy_placeholder)

    # When run, the init_op initialises any tensorflow variables
    # hint: weights and biases in our case
    init_op = tf.global_variables_initializer()

    # Start session
    sess = tf.Session()
    sess.run(init_op)


    # Initialise TensorBoard Summary writers
    dtstr = "{:%b_%d_%H-%M-%S}".format(datetime.now())
    train_writer = tf.summary.FileWriter('./summaries/' + tensorboard_name + '_' + dtstr + '/train', sess.graph)
    test_writer = tf.summary.FileWriter('./summaries/' + tensorboard_name + '_' + dtstr + '/test', sess.graph)

    # Train
    print('Starting Training...')
    print('main() argument, N: ', N_CLUSTER)

    train(sess, data, n_training_epochs, batch_size,
          summaries_op, accuracy_summary_op, train_writer, test_writer,
          X, Y, SIFT, train_op, loss_op, accuracy_op, N=N_CLUSTER)

    print('Training Complete')

    # Clean up
    sess.close()







import numpy as np
import os
import tensorflow as tf
import time

from datetime import datetime



class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        super().__init__(message)



def get_batch(data, batch_size):
	'''
	Function which returns a list of batch, each with batch_size data points. 
	'''
	indices = np.array(range(0,data[0].shape[0]))
	np.random.shuffle(indices)

	split = list(range(0,data[0].shape[0],batch_size))[1:]
	if data[0].shape[0] - split[-1] < batch_size:
		split.pop()
	split_indices = np.split(indices, split)

	batches = []

	for idx in split_indices:
		batches.append([data[0][idx,:], data[1][idx,:]])

	return batches
	


def split_data(data, ratio):
	'''
	Ratio is a list of number which represents the proportion of data to be put into trianing, testing and validation set.
	'''
	try:
		if len(ratio) not in range(2,4):
			raise InputError('The length of ratio must be 2 or 3')
		if sum(ratio) != 100 and sum(ratio) != 1:
			raise InputError('Ratio must sum up to 1 or 100')
		if len(data) != 2 or data[0].shape[0] != data[1].shape[0]:
			raise InputError('Input data must be of the form [X,Y] where X and Y have the same length.')
	except:
		raise
	else:
		indices = []
		k = 0

		for i in ratio[:-1]:
			k += i
			indices.append(k*data[0].shape[0])

		indices = np.array(indices)//sum(ratio)

		X_split = np.split(data[0], indices, axis=0)
		Y_split = np.split(data[1], indices, axis=0)

		return tuple(zip(X_split,Y_split))



def get_accuracy_op(preds_op, Y):
    with tf.name_scope('accuracy_ops'):
        correct_preds_op = tf.equal(tf.argmax(preds_op, 1), tf.argmax(Y, 1))
        # the tf.cast sets True to 1.0, and False to 0.0. With N predictions, of
        # which M are correct, the mean will be M/N, i.e. the accuracy
        accuracy_op = tf.reduce_mean(tf.cast(correct_preds_op, tf.float32))
    return accuracy_op



def accuracy(sess, data, batches, batch_size, X, Y, accuracy_op):
    # compute number of batches for given batch_size
    n_batches = len(batches)

    overall_accuracy = 0.0
    for i in range(n_batches):
        batch = batches[i]
        accuracy_batch = \
            sess.run(accuracy_op, feed_dict={X: batch[0], Y: batch[1]})
        overall_accuracy += accuracy_batch

    return overall_accuracy/n_batches



def convnet(X, Y, convlayer_sizes=[10, 10], filter_shape=[3, 3], outputsize=10, padding="same"):
    """
    Create a Tensorflow model for a Convolutional Neural Network. The network
    should be of the following structure:
    conv_layer1 -> conv_layer2 -> fully-connected -> output

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
    conv1: A convolutional layer of convlayer_sizes[0] filters of shape filter_shape
    conv2: A convolutional layer of convlayer_sizes[1] filters of shape filter_shape
    w: Connection weights for final layer
    b: biases for final layer
    logits: The inputs to the activation function
    preds: The outputs of the activation function (a probability
    distribution over the 10 digits)
    batch_xentropy: The cross-entropy loss for each image in the batch
    batch_loss: The average cross-entropy loss of the batch
    """
    #w1 = tf.Variable(tf.truncated_normal([filter_shape[0], filter_shape[0]], stddev=1.0/math.sqrt(float(X.shape[1])))) 
    
    conv1 = tf.layers.conv2d( X, filters=convlayer_sizes[0],
                         kernel_size=filter_shape[0],
                         padding=padding,
                         activation=tf.nn.relu, use_bias=True,
                         kernel_initializer=None)


    conv2 = tf.layers.conv2d( conv1, filters=convlayer_sizes[1],
                         kernel_size=filter_shape[1],
                         padding=padding,
                         activation=tf.nn.relu, use_bias=True,
                         kernel_initializer=None)

    dim = int(conv2.get_shape()[1]*conv2.get_shape()[2]*conv2.get_shape()[3])



    # Fully connected layer
    X = tf.reshape(conv2, [tf.shape(conv2)[0], dim])

    w = tf.Variable(tf.zeros([int(X.shape[1]), outputsize]))
    b = tf.Variable(tf.zeros([outputsize]))

    # Calculate the weighted sum 
    logits = tf.matmul(X, w) + b

    # preds will hold the predicted class
    preds = tf.nn.softmax(logits)

    # Cross entropy 
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=Y)
    batch_loss = tf.reduce_mean(batch_xentropy)
    
    return conv1, conv2, w, b, logits, preds, batch_xentropy, batch_loss



def train(sess, data, n_epochs, batch_size,
          summaries_op, accuracy_summary_op, train_writer, test_writer,
          X, Y, train_op, loss_op, accuracy_op, ratio=[70,20,10]):

    # record starting time
    train_start = time.time()

    train_data, test_data, validation_data = split_data(data, ratio)

    # Run through the entire dataset n_training_epochs times
    for i in range(n_epochs):
        # Initialise statistics
        training_loss = 0
        epoch_start = time.time()

        batches = get_batch(train_data, batch_size)
        n_batches = len(batches)

        # Run the SGD train op for each minibatch
        for i in range(n_batches):
            batch = batches[i]

            # Run a training step
            trainstep_result, batch_loss, summary = \
            	sess.run([train_op, loss_op, summaries_op], feed_dict={X: batch[0], Y: batch[1]})
            train_writer.add_summary(summary, i)
            training_loss += batch_loss

        # Timing and statistics
        epoch_duration = round(time.time() - epoch_start, 2)
        ave_train_loss = training_loss / n_batches

        # Get accuracy
        train_accuracy = \
            accuracy(sess, train_data, batches, batch_size, X, Y, accuracy_op)
        test_accuracy = \
            accuracy(sess, test_data, batches, batch_size, X, Y, accuracy_op)

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
        print('Epoch {0}, Training Loss: {1}, Test accuracy: {2}, \
time: {3}s, total time: {4}s'.format(i, ave_train_loss,
                                     test_accuracy, epoch_duration,
                                     train_duration))
    print('Total training time: {0}s'.format(train_duration))
    print('Confusion Matrix:')
    true_class=tf.argmax(Y, 1)
    predicted_class=tf.argmax(preds_op, 1)
    cm=tf.confusion_matrix(predicted_class,true_class)
    print(sess.run(cm, feed_dict={X: test_data[0],
                                  Y: test_data[1]}))






# ---------------------------------------------------------------------------- #

# from tensorflow.examples.tutorials.mnist import input_data
# dt = input_data.read_data_sets('data/mnist', one_hot=True)
# data = [dt.train.images, dt.train.labels]


import pandas as pd 

filepath = os.path.abspath(os.path.join('./Data/fer2013/', 'fer2013.csv'))

dt = pd.read_csv(filepath, sep=',', header=0)

train_dt = dt.loc[dt['Usage']=='Training',:]
validation_dt = dt.loc[dt['Usage']=='PrivateTest',:]
test_dt = dt.loc[dt['Usage']=='PublicTest',:]

# Convert labels into 1 hot encoding
labels = np.zeros((train_dt['emotion'].shape[0], 7))
labels[np.arange(labels.shape[0]), train_dt['emotion']] = 1

data = [np.array([i.split() for i in train_dt['pixels']], dtype=int), labels]

# ---------------------------------------------------------------------------- #


# data = np.array([list(dt.train.images), list(dt.train.labels)])
input_dim = data[0].shape[1]
output_dim = data[1].shape[1]

img_shape = [48,48]

tensorboard_name = 'fer2013'
# ---------------------------------------------------------------------------- #



n_training_epochs = 1
batch_size = 1000
learning_rate = 0.001


# Define dimension of input, X and output Y
X = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name="image_input")
Y = tf.placeholder(dtype=tf.float32, shape=[None, output_dim], name="image_target_onehot")


# Define the CNN
conv1out, conv2out, w, b, logits_op, preds_op, xentropy_op, loss_op = convnet(tf.reshape(X, [-1, img_shape[0], img_shape[1], 1]), Y, convlayer_sizes=[output_dim,output_dim], outputsize=output_dim)
tf.summary.histogram('pre_activations', logits_op)


# The training op performs a step of stochastic gradient descent on a minibatch
# optimizer = tf.train.GradientDescentOptimizer # vanilla SGD
# optimizer = tf.train.MomentumOptimizer # SGD with momentum
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
# hint: weights and biases in our case
init_op = tf.global_variables_initializer()

# Start session
sess = tf.Session()
sess.run(init_op)

# Initialise TensorBoard Summary writers
dtstr = "{:%b_%d_%H-%M-%S}".format(datetime.now())
train_writer = tf.summary.FileWriter('./summaries/'+tensorboard_name+'_'+dtstr+'/train', sess.graph)
test_writer  = tf.summary.FileWriter('./summaries/'+tensorboard_name+'_'+dtstr+'/test')

# Train
print('Starting Training...')

train(sess, data, n_training_epochs, batch_size,
      summaries_op, accuracy_summary_op, train_writer, test_writer,
      X, Y, train_op, loss_op, accuracy_op)

print('Training Complete')

# Clean up
sess.close()
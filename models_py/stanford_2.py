import tensorflow as tf

def NeuralNet(X, Y, SIFT, convlayer_sizes=[10, 10], filter_shape=[3, 3], outputsize=10, padding="same", SIFT_size = 2048):
    num_features = 32
    kernel = 5
    kernel_initializer = tf.truncated_normal_initializer()
    # bias_initializer = tf.truncated_normal_initializer()
    # kernel_initializer = None
    bias_initializer = None
    activation = tf.nn.leaky_relu
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)


    # --------------------------layer 1-----------------------------------
    layer1_conv1 = tf.layers.conv2d(X, filters=num_features,
                             kernel_size=kernel,
                             padding=padding,
                             activation=activation, 
                             use_bias=True,
                             bias_initializer=bias_initializer,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=regularizer)
    layer1_norm1 = tf.layers.batch_normalization(layer1_conv1)

    num_features = 64
    layer1_conv2 = tf.layers.conv2d(layer1_norm1, filters=num_features,
                             kernel_size=kernel,
                             padding=padding,
                             activation=activation, 
                             use_bias=True,
                             bias_initializer=bias_initializer,
                             kernel_initializer=kernel_initializer)
    layer1_norm2 = tf.layers.batch_normalization(layer1_conv2)

    layer1_maxPool1 = tf.layers.max_pooling2d(layer1_norm2,pool_size=(2,2),strides=(2,2))

    kernel = 3
    num_features = 128
    layer1_conv3 = tf.layers.conv2d(layer1_maxPool1, 
                            filters=num_features,
                             kernel_size=kernel,
                             padding=padding,
                             activation=activation, 
                             use_bias=True,
                             bias_initializer=bias_initializer,
                             kernel_initializer=kernel_initializer)

    layer1_dropout = tf.layers.dropout(layer1_conv3,rate=0.1)

    layer1_maxPool2 = tf.layers.max_pooling2d(layer1_dropout,pool_size=(2,2),strides=(2,2))

    dim = int(layer1_maxPool2.get_shape()[1] * layer1_maxPool2.get_shape()[2] * layer1_maxPool2.get_shape()[3])



    # -------------------------Fully connected layer ----------------------
    X = tf.reshape(layer1_maxPool2, [tf.shape(layer1_maxPool2)[0], dim])

    num_features = 2048
    FC1 = tf.layers.dense(X, num_features, activation=activation)
    FC1_drop = tf.layers.dropout(FC1, rate=0.5)

    num_features = 1024
    FC2 = tf.layers.dense(FC1_drop, num_features, activation=activation)
    FC2_drop = tf.layers.dropout(FC2, rate=0.5)

    num_features = 512
    FC3 = tf.layers.dense(FC2_drop, num_features, activation=activation)
    # FC3_drop = tf.layers.dropout(FC3, rate=0.5)

    # ------------------------- SIFT Layer 1 ------------------------------

    denseSIFT = tf.layers.dense(SIFT,num_features,
                                activation=activation, 
                                kernel_regularizer=regularizer
                                )
    # denseSIFT_drop = tf.layers.dropout(denseSIFT, rate=0.5)


    # ------------------------- Combined Layer ------------------------------
    dense_combined = tf.reduce_mean([denseSIFT, FC3], 0)  # [1.5, 1.5]
    dense_combined_drop = tf.layers.dropout(dense_combined, rate=0.5)


    logits = tf.layers.dense(inputs=dense_combined_drop,units=outputsize)


    # preds will hold the predicted class
    preds = tf.nn.softmax(logits)

    # Cross entropy
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    batch_loss = tf.reduce_mean(batch_xentropy)

    return logits, preds, batch_loss







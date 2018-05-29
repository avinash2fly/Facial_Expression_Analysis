
from FUNCTIONS import *

# ------------------------------------------------------------------------------------- #
# GLOBAL VARIABLES
# ------------------------------------------------------------------------------------- #

face_pred_path = "shape_predictor_68_face_landmarks.dat"
pickle_path = "./pickle_jar/pickle_objects.p"
model_path = "./models/model_complex_59/model.ckpt"

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


# ------------------------------------------------------------------------------------- #
# MAIN
# ------------------------------------------------------------------------------------- #

# if __name__ == "__main__":


#     # ------------------------ Read Images from file ------------------------ #

#     filepath = os.path.abspath(os.path.join('./Data/', 'CK_plus_fullsize.csv'))

#     dt = pd.read_csv(filepath, sep=',', header=0)

#     train_dt = dt.loc[dt['Usage'] == 'Training', :]

#     labels = np.zeros((train_dt['emotion'].shape[0], 7))
#     labels[np.arange(labels.shape[0]), train_dt['emotion']] = 1

#     real_labels = train_dt['emotion']

#     faces=[]
#     raw_faces=[]

#     for pixel_sequence in train_dt['pixels']:
#         # Processing from csv
#         image = np.array([int(pixel) for pixel in pixel_sequence.split(' ')], dtype=np.uint8)
#         image = image.reshape((490, 640))
#         processed_image, rects = preprocess(image, target_width=target_width)
#         faces += [img.flatten() for img in processed_image]
#         raw_faces+= [img for img in processed_image]
#         break

#     data =[np.array(faces)/255, labels]

#     input_dim = data[0].shape[1]
#     output_dim = data[1].shape[1]


#     # ------------------ Get SIFT Features ---------------------- #

#     # if N > SIFT_train.shape[0]:
#     #   N = int(sqrt(SIFT_train.shape[0]))

#     # Extract key points and descriptors
#     data_kp = extractKeyPoints(raw_faces)

#     # Get descriptors and get clusters
#     saved_params = pickle.load( open( "./pickle_jar/pickle_objects.p", "rb" ) )
#     clusterer = saved_params['clusterer']
#     N_CLUSTER = saved_params['N_CLUSTER']

#     # Get bag of features count for each image
#     SIFT_data = convert2bagOfFeatures(data_kp, clusterer, n_cluster = N_CLUSTER)

#     # Append result to train_data
#     data.append(SIFT_data/N_CLUSTER)


#     # ------------------ Read Model from Checkpoint ---------------------- #

#     # Define input placeholder
#     X = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name="image_input")
#     Y = tf.placeholder(dtype=tf.float32, shape=[None, output_dim], name="image_target_onehot")
#     SIFT = tf.placeholder(dtype=tf.float32, shape=[None, N_CLUSTER-1], name="SIFT_input")

#     # Define the CNN
#     logits_op, preds_op, loss_op = NeuralNet(tf.reshape(X, [-1, img_shape[0], img_shape[1], 1]), Y, SIFT, 
#         convlayer_sizes=[output_dim, output_dim], outputsize=output_dim, SIFT_size=2048)
#     tf.summary.histogram('pre_activations', logits_op)

#     # The training op performs a step of stochastic gradient descent on a minibatch
#     optimizer = tf.train.AdamOptimizer  
#     train_op = optimizer(learning_rate).minimize(loss_op)

#     # Prediction and accuracy ops
#     accuracy_op = get_accuracy_op(preds_op, Y)

#     # TensorBoard for visualisation
#     # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
#     summaries_op = tf.summary.merge_all()

#     # Separate accuracy summary so we can use train and test sets
#     accuracy_placeholder = tf.placeholder(shape=[], dtype=tf.float32)
#     accuracy_summary_op = tf.summary.scalar("accuracy", accuracy_placeholder)

#     # Global initializer
#     init_op = tf.global_variables_initializer()

#     # Start session
#     sess = tf.Session()

#     saver = tf.train.Saver()
#     saver.restore(sess, "./models/model.ckpt")

#     feed_dict = {X : data[0], Y : data[1], SIFT : data[2]}

#     _prediction = sess.run([preds_op], feed_dict)

#     prediction = [np.argmax(pred) for pred in _prediction]




# ------------------------ Read Images from webcam ------------------------ #
input_dim = 2304
output_dim = 3

# Get descriptors and get clusters
saved_params = pickle.load( open( pickle_path, "rb" ) )
clusterer = saved_params['clusterer']
N_CLUSTER = saved_params['N_CLUSTER']


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_pred_path)

# ------------------ Read Model from Checkpoint ---------------------- #

# Define input placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name="image_input")
Y = tf.placeholder(dtype=tf.float32, shape=[None, output_dim], name="image_target_onehot")
SIFT = tf.placeholder(dtype=tf.float32, shape=[None, N_CLUSTER-1], name="SIFT_input")

# Define the CNN
logits_op, preds_op, loss_op = NeuralNet(tf.reshape(X, [-1, img_shape[0], img_shape[1], 1]), Y, SIFT, 
    convlayer_sizes=[output_dim, output_dim], outputsize=output_dim, SIFT_size=2048)
tf.summary.histogram('pre_activations', logits_op)

# The training op performs a step of stochastic gradient descent on a minibatch
optimizer = tf.train.AdamOptimizer  
train_op = optimizer().minimize(loss_op)

# Prediction and accuracy ops
accuracy_op = get_accuracy_op(preds_op, Y)

# TensorBoard for visualisation
# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
summaries_op = tf.summary.merge_all()

# Separate accuracy summary so we can use train and test sets
accuracy_placeholder = tf.placeholder(shape=[], dtype=tf.float32)
accuracy_summary_op = tf.summary.scalar("accuracy", accuracy_placeholder)

# Global initializer
init_op = tf.global_variables_initializer()

# Start session
sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, model_path)

data_kp = None
SIFT_data = None

# img = cv2.imread("/Users/dian/Desktop/Multiple_1.png", 1)

cap = cv2.VideoCapture(0);
emotion = {0:'neutral', 1:'happy', 2:'sad'}


freeze = False
k = 0

while True:

    if not freeze:
        ret, img = cap.read()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if k == 13 : 
        freeze = not freeze
        print(freeze)

        if freeze:
            processed_image, rects = preprocess(image, detector, predictor, target_width=target_width)
            faces = [ img1.flatten() for img1 in processed_image]
            raw_faces = [ img1 for img1 in processed_image]

            data =[np.array(faces)/255]
            print(data)

            if len(data[0]) == 0:
                continue

            input_dim = data[0].shape[1]
            output_dim = 3


            # ------------------ Get SIFT Features ---------------------- #

            # if data_kp is None : 
            # Extract key points and descriptors
            data_kp = extractKeyPoints(raw_faces)
            
            # Get bag of features count for each image
            SIFT_data = convert2bagOfFeatures(data_kp, clusterer, n_cluster = N_CLUSTER)

            # Append result to train_data
            data.append(SIFT_data/N_CLUSTER)


            feed_dict = {X : data[0], SIFT : data[1]}

            _prediction = sess.run([preds_op], feed_dict)

            prediction = [np.argmax(pred) for pred in _prediction[0]]

            # rects, prediction, data_kp, SIFT_data = getPrediction(img, data_kp, 48, N_CLUSTER, SIFT_data)

            if rects is None:
                continue

            for rect, pred in zip(rects, prediction):
                print('Drawing Rects : ', rect)
                cv2.rectangle(img,(rect.left(),rect.bottom()),(rect.right(),rect.top()),(255,255,0),2)
                cv2.putText(img,emotion[int(pred)],(rect.left()+2,rect.bottom()+2), cv2.FONT_HERSHEY_SIMPLEX , 1,(255,255,0),2,cv2.LINE_AA)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()







from FUNCTIONS import *

# ------------------------------------------------------------------------------------- #
# GLOBAL VARIABLES
# ------------------------------------------------------------------------------------- #

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


# ------------------------------------------------------------------------------------- #
# FUNCTIONS
# ------------------------------------------------------------------------------------- #


# def detect_facial_landmarks(img, detector, predictor, draw=False, drawIdx = None):
#     landmarks = []
#     faces = []

#     # Detect face region using dlib
#     rects = detector(img, 0)
     
#     # loop over each face detected
#     for (i, rect) in enumerate(rects):

#         # Detect facial landmarks in the given face region
#         points = predictor(img, rect)
#         points = face_utils.shape_to_np(points)

#         faces.append(rect)
#         landmarks.append(points)

#         # Draw circle for each facial landmarks detected
#         if draw:
#             if drawIdx is not None:
#                 drawPoints = points[drawIdx]
#             else:
#                 drawPoints = points
#             for (x, y) in drawPoints:
#                 cv2.circle(img, (x, y), 2, (255, 255, 0), -1)

#     return faces, landmarks





# ------------------------------------------------------------------------------------- #
#  IMAGE PROCESSING
# ------------------------------------------------------------------------------------- #

# def preprocess(image, zoom=0, target_width=None, LEFT_EYE_IDXS=LEFT_EYE_IDXS, RIGHT_EYE_IDXS=RIGHT_EYE_IDXS):
#     '''
#     This function does:
#     1. Histogram equalization
#     2. Face alignment
#     3. Crop face into image of size (target_width,target_width)
#     '''

#     target_left_eye_pos = [zoom, 0.35]   # [x,y] coordinate
#     target_right_eye_pos = [1-zoom, 0.35]

#     target_height = target_width

#     if target_width is None:
#         target_width = image.shape[1]
#         target_height = image.shape[0]

#     target_distance = target_width*(target_right_eye_pos[0] - target_left_eye_pos[0])

#     p = "shape_predictor_68_face_landmarks.dat"
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor(p)
    
#     if len(image.shape) > 2:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     processed_images = []

#     # 1 Histogram equalization

#     img = cv2.equalizeHist(image)

#     # 2 Detect face landmarks using dlib
#     faces, landmarks = detect_facial_landmarks(img, detector, predictor, draw=False, drawIdx=LEFT_EYE_IDXS+RIGHT_EYE_IDXS)

#     for face, landmark in zip(faces, landmarks):

#         # 3 Calculate the centre of left and right eyes
#         left_eye = landmark[LEFT_EYE_IDXS,]
#         right_eye = landmark[RIGHT_EYE_IDXS,]
#         left_centre = np.mean(left_eye,axis=0)
#         right_centre = np.mean(right_eye,axis=0)
#         slope = right_centre - left_centre

#         # 4 Get scaling factor by taking the ratio of the target distance between the centre of
#         #     the left and right eyes with the actual distance between the left and right eyes. 
#         distance = np.sqrt((slope[0] ** 2) + (slope[1] ** 2))
#         scale = target_distance / distance
#         shift_x = target_width
#         shift_y = target_height
#         if zoom == 0:
#             scale = 1
#             shift_x = img.shape[1]
#             shift_y = img.shape[0]

#         # 5 Calculate the angle 
#         angle = np.degrees(np.arctan2(slope[1], slope[0]))

#         # 6 Get rotation matrix based on the calculated scaling factor and angle
#         pivot = (left_centre+right_centre)//2
#         M = cv2.getRotationMatrix2D(tuple(pivot.astype(int)), angle, scale)

#         # 7 Update transalation part of the rotation matrix
#         tX = shift_x * 0.5
#         tY = shift_y * target_left_eye_pos[1]
#         M[0, 2] += (tX - pivot[0])
#         M[1, 2] += (tY - pivot[1])

#         # # Draw centre of left eye
#         # cv2.circle(img, tuple(left_centre.astype(int)), 2, (255, 255, 0), -1)
#         # # Draw centre of right eye
#         # cv2.circle(img, tuple(right_centre.astype(int)), 2, (255, 255, 0), -1)
#         # # Draw centre of rotation
#         # cv2.circle(img, tuple(pivot.astype(int)), 2, (255, 255, 0), -1)
#         # # Draw a line between eyes
#         # cv2.line(img,tuple(left_centre.astype(int)),tuple(right_centre.astype(int)),(255,255,0),2)
#         # # Draw square
#         # cv2.rectangle(img,(face.left(),face.bottom()),(face.right(),face.top()),(255,255,0),2)

#         # apply the affine transformation
#         final_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

#         rects = detector(final_img, 0)

#         # Detect face again on the now aligned image
#         for rect in rects:
#             cropped_face = final_img[rect.top():rect.bottom(), rect.left():rect.right()]
#             print('target widht : ', target_width)
#             cropped_face = cv2.resize(cropped_face, (target_width, target_width))
#             processed_images.append(cropped_face)

#     return processed_images



# def plot(img, final_img):
#     fig = plt.figure()
#     fig.subplots_adjust(hspace=0.3)
#     plt.subplot(1, 2, 1)
#     plt.imshow(img)
#     plt.subplot(1, 2, 2)
#     plt.imshow(final_img)
#     plt.show()



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
#     clusterer = getCluster(data_kp, n_cluster=N_CLUSTER)
    
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

cap = cv2.VideoCapture(0);
emotion = {0:'neutral', 1:'anger', 2:'contempt', 3:'disgust', 4:'fear', 5:'happy', 6:'sadness', 7:'surprise'}

while True:
    ret, img = cap.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    processed_image, rects = preprocess(image, target_width=target_width)
    faces = [img.flatten() for img in processed_image]
    raw_faces = [img for img in processed_image]

    data =[np.array(faces)/255]
    print(data)

    if len(data[0]) == 0:
        continue
        
    input_dim = data[0].shape[1]
    output_dim = 7


    # ------------------ Get SIFT Features ---------------------- #

    # if N > SIFT_train.shape[0]:
    #   N = int(sqrt(SIFT_train.shape[0]))

    # Extract key points and descriptors
    data_kp = extractKeyPoints(raw_faces)

    # Get descriptors and get clusters
    clusterer = getCluster(data_kp, n_cluster=N_CLUSTER)
    
    # Get bag of features count for each image
    SIFT_data = convert2bagOfFeatures(data_kp, clusterer, n_cluster = N_CLUSTER)

    # Append result to train_data
    data.append(SIFT_data/N_CLUSTER)


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
    train_op = optimizer(learning_rate).minimize(loss_op)

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
    saver.restore(sess, "./models/model.ckpt")

    feed_dict = {X : data[0], Y : data[1], SIFT : data[2]}

    _prediction = sess.run([preds_op], feed_dict)

    prediction = [np.argmax(pred) for pred in _prediction]

    for rect, pred in zip(rects, prediction):
        cv2.rectangle(image,(rect.left(),rect.bottom()),(rect.right(),rect.top()),(255,255,0),2)
        cv2.putText(image,emotion[int(pred)],(rect.left()+2,rect.bottom()+2), cv2.FONT_HERSHEY_SIMPLEX , 1,(255,255,0),2,cv2.LINE_AA)

    cv2.imshow('img', image)
    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()





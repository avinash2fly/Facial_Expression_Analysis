import cv2
import dlib 
import numpy as np
import pandas as pd 
import os 

from FUNCTIONS import *

filepath = os.path.abspath(os.path.join('./Data/fer2013/', 'fer2013.csv'))
filepath_ck = os.path.abspath(os.path.join('./Data/', 'CK_plus_48.csv'))

output_file = './Data/CK_fer_data.csv'
# emotion = {0:'neutral', 1:'anger', 2:'contempt', 3:'disgust', 4:'fear', 5:'happy', 6:'sadness', 7:'surprise'}
# (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)

ck_2_fer = {0:6, 1:0, 3:1, 4:2, 5:3, 6:4, 7:5}

# Reading fer dataset
dt = pd.read_csv(filepath, sep=',', header=0)
dt_ck = pd.read_csv(filepath_ck, sep=',', header=0)

# grouped = dt_combine.groupby('emotion')
# grouped.count()

# fer2013 data count.       # CK data count
#          pixels  Usage.   #          pixels  Usage
# emotion                   # emotion               
# 0          4953   4953.   # 0            45     45
# 1           547    547.   # 1            18     18
# 2          5121   5121.   # 2            59     59
# 3          8989   8989.   # 3            25     25
# 4          6077   6077.   # 4            69     69
# 5          4002   4002.   # 5            28     28
# 6          6198   6198.   # 6            83     83

#          Usage  pixels
# emotion               
# 0         4971    Angry
# 1          572    Disgust
# 2         5190    Fear

# 3         9018    Happy
# 4         6160    Sad

# 5         4002    Surprise

# 6         6243    Neutral

# Remap emotion label in CK+ data to match fer2013 definition
drop_idx = []
for i in range(dt_ck.shape[0]):
    if dt_ck.loc[i, 'emotion'] == 2:
        drop_idx.append(i)
        continue
    dt_ck.loc[i, 'emotion'] = ck_2_fer[dt_ck.loc[i, 'emotion']]

dt_ck = dt_ck.drop(drop_idx)

dt_combine = pd.concat([dt, dt_ck])



count = { 3:0, 4:0, 6:0}

idx_map = {6:0, 3:1, 4:2}

max_cnt = 600

final_dt = []


p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

path = 'haarcascade/'
face_cascade = cv2.CascadeClassifier(path+'haarcascade_frontalface_default.xml')

for i, row in dt_combine.iterrows():
    if row['emotion'] not in count:
        continue

    img = np.array(row['pixels'].split(' '), dtype=np.uint8).reshape((48,48))
    img = cv2.equalizeHist(img)
    # faces, landmarks = detect_facial_landmarks(img, detector, predictor, draw=False)
    faces = face_cascade.detectMultiScale(img,1.3,2)
    if len(faces) == 0:
        continue
    if count[row['emotion']] == max_cnt:
        continue
    final_dt.append([row['pixels'], idx_map[row['emotion']]])
    count[row['emotion']] += 1


final_dt = np.array(final_dt)

print(final_dt.shape)



output = open(output_file, 'w')
output.write('pixels,emotion, Usage\n')

for row in final_dt:
    output.write(row[0]+','+row[1]+',Training\n')

output.close()

# select_indices = np.where( np.logical_or(final_dt[:,1]=='3', final_dt[:,1]=='4', final_dt[:,1]=='6') )[0]

# label_3 = final_dt[final_dt[:,1]=='3',:600]
# label_4 = final_dt[final_dt[:,1]=='4',:600]

# label_6 = final_dt[final_dt[:,1]=='6',:600]








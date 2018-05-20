import cv2
import numpy as np 
import os
import pandas as pd 

import matplotlib.pyplot as plt


filepath = os.path.abspath(os.path.join('./Data/fer2013/', 'fer2013.csv'))

data = pd.read_csv(filepath, sep=',', header=0)

train = data.loc[data['Usage']=='Training',:]
validation = data.loc[data['Usage']=='PrivateTest',:]
test = data.loc[data['Usage']=='PublicTest',:]



# ---------------------------------------------------------------- #
#                      SINGLE IMAGE PROCESSING
# ---------------------------------------------------------------- #
img = np.array(train['pixels'].loc[0].split(), dtype=int)
label = np.array(train['emotion'].loc[0])
reshape_img = np.reshape(img, shape=(48,48))

# # Blur image to filter out noises - fer2013 images is probably too small to be blurred.
# img = cv2.GaussianBlur( img ,(3,3), 0)



plt.imshow(reshape_img, cmap='gray')
plt.show()

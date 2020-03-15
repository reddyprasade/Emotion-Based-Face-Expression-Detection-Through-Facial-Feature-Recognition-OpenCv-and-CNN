import os
import cv2
import keras
import pickle
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras import backend as K
from keras import regularizers
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
path = './project/'
##
### 1-hot encoding
a = np.array([i for i in range(len(os.listdir(path)))])
classes = np.zeros((a.size, a.max() + 1))
classes[np.arange(a.size), a] = 1
face_cascade = cv2.CascadeClassifier('Data/haarcascade_frontalface_default.xml')
train_array = []
test_array = []
dir_array = []
for dir in os.listdir(path):
    i1, i2 = 8, randint(0, 47)
    for idx, img in enumerate(sorted(os.listdir(path + dir))):
        image = cv2.imread(path + dir + '/' + img, 0)
        #cv2.imshow('frame',image)
        frame = cv2.resize(image,(640,480))
        #im1 = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        face=face_cascade.detectMultiScale(frame)
        for x,y,w,h in (face):
            cv2.rectangle(frame,(x,y),(x+w,y+h),[255,0,0],4)
            im_f = frame[y:y+h,x:x+w]
            im_f = cv2.resize(im_f,(112,92))
        image = cv2.resize(im_f, (32, 32))
        image = image[:, :, np.newaxis]
        if idx == i1 or idx == i2:
            
            test_array.append((image, classes[os.listdir(path).index(dir)]))
            continue

        train_array.append((image, classes[os.listdir(path).index(dir)]))
    dir_array.append(dir)
##    
input_shape = (32, 32, 1)

model=Sequential()

# convolutional layer 16 windows/filters of 3x3
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 kernel_regularizer=regularizers.l2(0.),
                 activity_regularizer=regularizers.l2(0.),
                 input_shape=input_shape))
# max of each 2x2 block
model.add(MaxPooling2D(pool_size=(2, 2)))
# normalization
model.add(BatchNormalization())
# avoid overfitting
model.add(Dropout(0.25))
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_regularizer=regularizers.l2(0.),
                 activity_regularizer=regularizers.l2(0.)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
# flatten for final layers
model.add(Flatten())
# fully-connected layer
model.add(Dense(3000, activation='relu',
                kernel_regularizer=regularizers.l2(0.),
                activity_regularizer=regularizers.l2(0.)))

model.add(Dropout(0.25))

model.add(Dense(len(os.listdir(path)), activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_images, train_labels = np.array([t[0] for t in train_array]), np.array([t[1] for t in train_array])
test_images, test_labels = np.array([t[0] for t in test_array]), np.array([t[1] for t in test_array])

history = model.fit(train_images, train_labels,
                    batch_size=20,
                    epochs=30,
                    verbose=2,
                    validation_data=(test_images, test_labels))

with open('Model/emotion_recognition_model2.pkl', 'wb') as f:
    pickle.dump(model, f)

###return test_images, test_labels, classes, dir_array
with open('Model/emotion_recognition_model2.pkl', 'rb') as f:
    model = pickle.load(f)
##
score = model.evaluate(test_images, test_labels, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

##
##
##
##

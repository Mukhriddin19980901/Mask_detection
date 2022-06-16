#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import cv2
import cvlib as cv
import glob
import os
tf.config.list_physical_devices()


# In[12]:


# step 1
def dataset(path):
    t=0
    x_train = []
    y_train = []
    for  i in range(3):
        labels = ['incorrect_mask','with_mask','without_mask']
        path_new = path + labels[i] + '/*'
        for image in glob.glob(path_new):
            image = cv2.imread(image)
            t+=1
            image = cv2.resize(image,(128,128))
            x_train.append(image)
            if i==0:
                y_train.append([1,0,0])
            elif i==1:
                y_train.append([0,1,0])
            else:
                y_train.append([0,0,1])
    return x_train,y_train
train_dir = r'datasets/maska/train/'
test_dir = r'datasets/maska/test/'

#step 2
x_train = []
y_train = []
x_train,y_train= dataset(train_dir)
x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape, y_train.shape)
x_test = []
y_test = []
x_test,y_test= dataset(test_dir)
x_test = np.array(x_test)
y_test = np.array(y_test)
print(x_test.shape)


# In[13]:


model = keras.Sequential([
    keras.layers.Conv2D(32,(3,3) ,activation='relu' ,input_shape=(128,128,3)),
    keras.layers.MaxPooling2D((3,3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64,(3,3),activation = 'relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128,(3,3),activation = 'relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation = 'relu'),
    keras.layers.Dense(64,activation = 'relu'),
    keras.layers.Dense(32,activation = 'relu'),
    
    keras.layers.Dense(3,activation = 'softmax'),
])
optim = keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=optim,loss='categorical_crossentropy',metrics=['accuracy'])
with tf.device("/GPU:0"):
    model2=model
    model2.fit(x_train,y_train,epochs=10)
    model2.evaluate(x_test,y_test)
    model2.summary()


# In[26]:


model2.save('maska_detection.model',save_format='h5')


# In[7]:


model2 = keras.models.load_model('maska_detection.model')


# In[8]:


inp = "video_adress_here"
video = cv2.VideoCapture(inp)
video.set(3,1640) 
video.set(4,1480)
faces = ['wrong_mask','mask','no_mask']

while video.isOpened():
    _,image = video.read()
    face,confidence = cv.detect_face(image)
    for (x,y,w,h) in face:
        yuz_np = np.copy(image[y:h,x:w])
        if yuz_np.shape[0]<10 or yuz_np.shape[1]<10:
            continue
        yuz_np = cv2.resize(yuz_np,(128,128))
        yuz_np = np.expand_dims(yuz_np,0)
        bashorat = model2.predict(yuz_np)
        index=np.argmax([bashorat])
        face = faces[index]
    
        if face=='mask':
            color = (0,255,0)
        elif face== "wrong_mask" : 
            color = (255,130,0)
        else:
            color = (0,0,255)
        if index:
            face = f'{face} :  {np.around((1-bashorat[0][0])*100,2)} %'
        else:
            face = f'{face} : {np.around((bashorat[0][0]*100),2)} %'
        if y-10>10:
            Y=y-10
        else:
            Y=y+10        
        cv2.rectangle(image,(x,y),(w,h),color,2)
        cv2.putText(image,face,(x,Y),cv2.FONT_HERSHEY_COMPLEX,0.7,color,2)
            
    cv2.imshow("mask_detection",image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video.release()
cv2.destroyAllWindows()


# In[22]:





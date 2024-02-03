###               Mask Detection using Convolutional Neural Network
<img src="https://github.com/Mukhriddin19980901/Mask_detection/blob/main/pics/maskgif.gif?raw=true" width="500" height="500" />

This repository contains [code](https://github.com/Mukhriddin19980901/Mask_detection/blob/main/face_mask_project.ipynb) for a mask detection system using a Convolutional Neural Network (CNN). The system is designed to classify images into three categories: incorrect mask, mask, and no mask.     
# Table of Contents
- Dataset
- Data Preprocessing
- Model Architecture
- Model Training
- Testing
 # Dataset
The dataset consists of images belonging to three classes: incorrect mask, mask, and no mask. The images are loaded and preprocessed using OpenCV and TensorFlow.   

    import tensorflow as tf
    import cv2
    import numpy as np
    import cvlib as cv
    import glob

    def dataset(path):
        # Function to load and preprocess the dataset
        # ...
         return x_train, y_train

    # Example usage:
     train_dir = r'datasets/maska/train/'
     test_dir = r'datasets/maska/test/'
     x_train, y_train = dataset(train_dir)
     x_test, y_test = dataset(test_dir)

# Data Preprocessing
Images are resized to 128x128 pixels and loaded into NumPy arrays for training and testing.
 import numpy as np

   # Data preprocessing
   x_train = np.array(x_train)
   y_train = np.array(y_train)
   print(x_train.shape, y_train.shape)

   x_test = np.array(x_test)
   y_test = np.array(y_test)
   print(x_test.shape)
   
# Model Architecture
The model is a sequential CNN with convolutional, max-pooling, batch normalization, and dense layers.
   from tensorflow import keras

   model = keras.Sequential([
       # Model layers...
   ])

   # Model compilation
   optim = keras.optimizers.Adam(learning_rate=0.001)
   model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
# Model Training
The model is trained on the preprocessed dataset using TensorFlow. GPU acceleration is utilized if available.

   with tf.device("/GPU:0"):
       model.fit(x_train, y_train, epochs=10)
       model.evaluate(x_test, y_test)
       model.summary()

# Testing
The trained model is saved and then loaded for real-time testing using a webcam.

   # Example usage for real-time testing
   model.save('mask_detection_model.h5', save_format='h5')

   # Load the saved model
   model = keras.models.load_model('mask_detection_model.h5')

   # Real-time testing using webcam
   # ...

<img src="https://github.com/Mukhriddin19980901/Mask_detection/blob/main/pics/no_masks.png" width="300" height="300" /><img src="https://github.com/Mukhriddin19980901/Mask_detection/blob/main/pics/mask.png" width="300" height="300" /><img src="https://github.com/Mukhriddin19980901/Mask_detection/blob/main/pics/wrong_mask.png" width="300" height="300" /> 

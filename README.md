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


<img src="https://github.com/Mukhriddin19980901/Mask_detection/blob/main/pics/no_masks.png" width="300" height="300" /><img src="https://github.com/Mukhriddin19980901/Mask_detection/blob/main/pics/mask.png" width="300" height="300" /><img src="https://github.com/Mukhriddin19980901/Mask_detection/blob/main/pics/wrong_mask.png" width="300" height="300" /> 

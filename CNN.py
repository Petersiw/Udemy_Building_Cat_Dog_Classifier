#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:32:59 2017

@author: Petersiw
"""
#Building the CNN

#Importing the Keras libraries and packages
from keras.models import Sequential     #sequential neural network
from keras.layers import Convolution2D  #2D for images;3D for videos
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import ZeroPadding2D

#Initialising the CNN
classifier = Sequential()

#Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (128, 128, 3), 
                             activation = 'relu')) 
#64, 64, 3 for tensorflow 
                             
#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Adding one more convolution layer
classifier.add(ZeroPadding2D())
classifier.add(Convolution2D(64, (3, 3), activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#One more convolution layer
classifier.add(ZeroPadding2D())
classifier.add(Convolution2D(128, (3, 3), activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full connection
#hidden layer - 128 nodes
classifier.add(Dense(units = 256, activation = 'relu'))

#one more hidden layer - 
classifier.add(Dense(units = 512, activation = 'relu'))

#output layer - 1 node/sigmoid for binary outcomes/softmax for multiple outcomes
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Compiling the CNN
#categorical_crossentropy loss function for multiple class
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   width_shift_range = 0.2, 
                                   height_shift_range = 0.2)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000/32,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000/32,
                         use_multiprocessing = True)

#one convolution and hidden layer = acc:0.8114/val. acc:0.7720
#two convolution and one hidden layers = acc:0.8881/val. acc:0.8125
#two convolution and hidden layers(128/256 nodes) = acc:0.8934/val. acc:0.8115
#two convolution and hidden layers(256/256 nodes) = acc:0.9098/val. acc:0.7870
#two convolution and hidden layers(256/512 nodes) with 128 pix = acc:0.7995/val. acc:0.8085

#Making prediction on CNN
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', 
                            target_size = (64, 64))
test_image = image.img_to_array(test_image)     #colour dimension
test_image = np.expand_dims(test_image, axis = 0) #batch dimension
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'





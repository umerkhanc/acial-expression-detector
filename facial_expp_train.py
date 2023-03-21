# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 15:21:39 2020

@author: DELL
"""

from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Activation, Flatten,BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
from tensorflow.keras.optimizers import RMSprop,SGD,Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

num_classes=5 #mention number of classes
img_row,img_col=48,48 #img size
batch_size=8 # no. of images to train at a time


#set the path
train_dir=r'images\train'
validation_dir=r'images\validation'

#create a dataset from existing dataset by rescale,rotation, zoom,shift,flip
traindata_gen=ImageDataGenerator(rescale=1./255, rotation_range=30, shear_range=0.3,
                            zoom_range=0.3, width_shift_range=0.4,
                            height_shift_range=0.4,horizontal_flip=True,vertical_flip=True,
                            fill_mode='nearest')
valdata_gen=ImageDataGenerator(rescale=1./255)

#add parameters to the dataset created by ImageDataGenerator to create final datasets
train_data=traindata_gen.flow_from_directory(train_dir, target_size=(img_row,img_col),
                                             batch_size=batch_size,color_mode='grayscale',
                                             class_mode='categorical',
                                             shuffle=True)
validation_data=valdata_gen.flow_from_directory(validation_dir,target_size=(img_row,img_col),
                                                batch_size=batch_size,color_mode='grayscale',
                                                class_mode='categorical',
                                                shuffle=True)

model=Sequential()

# block1
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_row,img_col,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_row,img_col,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block2
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#bolck3
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#bolck4
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block5
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#bolck6 
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#block7
model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())


#%%
checkpoint= ModelCheckpoint('expression_vgg.h5',
                            monitor='val_loss',
                            mode='min',
                            save_best_only=True,
                            verbose=1)
earlystp=EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=10,
                       verbose=1,
                       restore_best_weights=True)
lr=ReduceLROnPlateau(monitor='val_loss',
                     factor=0.2,
                     patience=10,
                     verbose=1,
                     min_delta=0.0001)
callbacks=[earlystp,checkpoint,lr]
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
train_sample=24282
val_sample=5937
epochs=25
history=model.fit_generator(train_data,
                            steps_per_epoch=train_sample//batch_size,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=validation_data,
                            validation_steps=val_sample//batch_size)

















































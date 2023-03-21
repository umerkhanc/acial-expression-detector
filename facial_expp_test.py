# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 18:17:20 2020

@author: DELL
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np


#importing face recognizer algorithm -> har_casecade and trained model -> expression_vgg
face_expp=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model=load_model('expression_vgg.h5')


#accessing camera and taking frame as test input
clas_name=['Angry','Happy','Sad','Neutral','Surprise']
cp=cv2.VideoCapture(0)
while True:
    ret,frame=cp.read() # take input
    gray_sc=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #converting RGB to gray
    faces=face_expp.detectMultiScale(gray_sc,1.3,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3) #creating rectangle around the face
        roi_gray=gray_sc[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)
            
        #prediction
            pred=model.predict(roi)[0]
            label=clas_name[pred.argmax()]
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            
        else:
            cv2.putText(frame,'NO FACE FOUND',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cp.release()
cv2.destroyAllWindows()

    
            
        
        
        
    






























import sys
import cv2
import numpy as np
import threading
import time
import queue
#import paho.mqtt.client as mqtt
import os
os.environ[ 'MPLCONFIGDIR' ] = '/tmp'
os.chdir('/home/jh/w210_capstone/image_capture/GUI')
from mplwidget import MplWidget
from gui2021 import *
import json
import PIL.Image, PIL.ImageDraw
import base64
import uuid
from datetime import datetime
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
"""Modification of detect.py that should run without args and generate a list of images of persons"""
import argparse
import os
import platform
import shutil
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import face_recognition
import glob
import traceback
from s3_boto import retrive_all_face_keys, retrive_data_by_key, save_face_data
import cv2
import uuid
import base64
import numpy as np
import pandas as pd
from s3_boto import *

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def gstreamer_pipeline(
    # capture_width=4032,
    # capture_height=3040,
    capture_width=2016,
    capture_height=1520,
    display_width=640,
    display_height=480,
    # display_width=4032,
    # display_height=3040,
    framerate=10,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def recognit_face(new_img):
    #Needs to be updated
    print("start finding face")
    try:
        encoding_list = []
        existed_face__key_list = retrive_all_face_keys()
        #new_img = cv2.imdecode(new_img, cv2.COLOR_BGR2RGB)
        new_face_encoding = face_recognition.face_encodings(new_img)
        for face_img_key in existed_face__key_list:
            face_img_object = retrive_data_by_key(face_img_key)
            face_img = np.asarray(bytearray(face_img_object), dtype="uint8")
            face_img = cv2.imdecode(face_img, cv2.COLOR_BGR2RGB)
            face_img_encoding = face_recognition.face_encodings(face_img)
            encoding_list.append(face_img_encoding[0])
        if len(new_face_encoding)>0:
            result_list = face_recognition.compare_faces(encoding_list, new_face_encoding[0])
            find_face = [i for i, x in enumerate(result_list) if x]
        face_id = str(uuid.uuid4())
        print("result_list ")
        print(result_list)
        print("find_face ")
        print(find_face)
        if len(find_face) == 0: ## new face
            print("new face " + face_id)
        else:
            index = find_face[0]
            face_id = existed_face__key_list[index]
            face_id = face_id.split('/')[1].split('.')[0]
            print("existed face " + face_id)
        save_face_data(new_img, face_id)
        print("face id " + face_id)
        return face_id
    except Exception:
        print("recognit_face")
        traceback.print_exc() 

def crop_and_show_face(frame):
        faces=[]
        face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
        #convert to gray scale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray=cv.equalizeHist(gray) #it seems this is not needed
        #detect faces
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        if faces is not None:
            faceROI=np.zeros([480,640,3])
            for (x,y,w,h) in faces:
                center = (x + w//2, y + h//2)
                faceROI = frame[max([0,int(y-0.4*h)]):int(y+h+0.2*h),max([0,int(x-0.2*w)]):int(x+w+0.2*w)]
            return faceROI
        else:
            return None

def aggregate_object(user_object):
  img=user_object[0]
  face_id = recognit_face(img)
  print("got face id, start updating db")
  get_and_insert_user_data(user_object, face_id)
  historical_data = {}
  print("finished updating, start getting historical data")
  historical_data['history'] = retrive_user_hitstorical_data_by_face_id(face_id)
  historical_data['session-id'] = user_object['session-id']
  print('DB updated')

if __name__ == "__main__":
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            face=crop_and_show_face(img).astype(np.uint8)
            if isinstance(face,np.ndarray):
                if np.max(face.flatten())>0:
                    cv2.imshow("CSI Camera", face)
            #get ID of face
                #new_face_encoding = face_recognition.face_encodings(face)
                #print('nfe:',new_face_encoding)
                    recognit_face(face)
            else:
                print('type:',type(face))
            # This also acts as
            keyCode = cv2.waitKey(100) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

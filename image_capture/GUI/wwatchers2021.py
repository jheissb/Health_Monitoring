#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" wwtchers.py opens a GUI to interact with user. 
Face and body pictures are taken and messages are sent to the respective containers for BMI and pose estimation. 
If detection is OK, data is sent to the cloud for face recognition and historic data is returned. 
The new data is appended to the database in S3. Program loops every 20 ms cheking 
if new frames need to be displayed and messages need to be sent"""
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
OUTPUT_MODEL_DIR='/home/jh/w210_capstone/image_capture/GUI/models'
OUTPUT_MODEL_NAME='bmi_RF.model'
from mplwidget import MplWidget
from gui2021 import *
import json
import PIL.Image, PIL.ImageDraw
import base64
import uuid
from datetime import datetime
import matplotlib
matplotlib.use('Qt5Agg')
from pose_detector import detect_pose
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
from s3_boto import retrive_all_face_keys, retrive_data_by_key, save_face_data,get_and_insert_user_data,remove_oneliners
import cv2
import uuid
import base64
import numpy as np
import pandas as pd
from utility import load_model
#from face2bmi import *
#from models import predict_bmi
from yolo5.models.experimental import attempt_load
from yolo5.utils.datasets import LoadStreams, LoadImages
from yolo5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, 
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from yolo5.utils.torch_utils import select_device, load_classifier, time_synchronized
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
class opt:
    out='inference/output'
    source='0'
    weights='yolov5m.pt'
    view_img=False
    save_txt=False
    imgsz = 640
    webcam = True
    conf_thres=0.5
    iou_thres=0.5
    classes=None
    agnostic_nms=False

def draw_point(x,y,img):
    thickness = 5
    height, width, channel = img.shape
    draw = PIL.ImageDraw.Draw(img)
    draw.line([ int(x)-1,int(y),int(x)+1,int(y)],width = thickness,fill=(51,51,204))
    for xi in range (thickness):
        for yi in range (thickness):
            img[int(x)-xi+2,int(y)-yi+2]=200
    img[int(x)-1,int(y)]=0
    img[int(x)+1,int(y)]=0
    img[int(x),int(y)]=0
    img[int(x),int(y)-1]=0
    img[int(x),int(y)+1]=0    
    return(img)

def gstreamer_pipeline(
    # capture_width=4032,
    # capture_height=3040,
    capture_width=1600,#1080,#1920/2,#int(4032/2),
    capture_height=1400,#/2,#int(3040/2),
    # display_width=640,
    # display_height=480,
    display_width=1600,#1920/2,#int(4032/2),
    display_height=1400,#/2,#int(3040/2),
    # display_width=4032,
    # display_height=3040,
    framerate=30,
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

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



def pdist(a,b):
    #calculate euclidean distance between x and y
    return(((a[1]-b[1])**2+(a[0]-b[0])**2))**0.5

def calculate_ratio(img, keypoints):
    w, h = img.size
    height=0.1
    #height is calculated as nose->neck->max([rshoulder->rhip->rknee->rankle,lshoulder->lhip->lknee->lankle)
    if all(keypoints[0][1:]) and all(keypoints[17]):
        nose_points=np.asarray([round(keypoints[0][2] * w), round(keypoints[0][1] * h)])
        neck_points=np.asarray([round(keypoints[17][2] * w), round(keypoints[17][1] * h)])
        height+=pdist(nose_points,neck_points)*2#factor of two to account for nose-hair and ankle floor distances
        rheight=0
        lheight=0
        #right side:
        if all(keypoints[12]) and all(keypoints[6]) and all(keypoints[14]) and all(keypoints[16]):
            right_hip_points = np.asarray([round(keypoints[12][2] * w), round(keypoints[12][1] * h)]) 
            right_shoulder_points = np.asarray([round(keypoints[6][2] * w), round(keypoints[6][1] * h)])
            right_knee_points = np.asarray([round(keypoints[14][2] * w), round(keypoints[14][1] * h)])
            right_ankle_points = np.asarray([round(keypoints[16][2] * w), round(keypoints[16][1] * h)])
            rheight=pdist(right_ankle_points,right_knee_points)+pdist(right_hip_points,right_knee_points)+\
                pdist(right_hip_points,right_shoulder_points)
        #left side
        if all(keypoints[11]) and all(keypoints[5]) and all(keypoints[15]) and all(keypoints[13]) and all(keypoints[14]):
            left_hip_points = np.asarray([round(keypoints[11][2] * w), round(keypoints[11][1] * h)]) 
            left_shoulder_points = np.asarray([round(keypoints[5][2] * w), round(keypoints[5][1] * h)])
            left_knee_points = np.asarray([round(keypoints[13][2] * w), round(keypoints[13][1] * h)])
            left_ankle_points = np.asarray([round(keypoints[15][2] * w), round(keypoints[15][1] * h)])
            lheight=pdist(left_ankle_points,left_knee_points)+pdist(left_hip_points,left_knee_points)+\
                pdist(left_hip_points,left_shoulder_points)
        height+=np.max([lheight,rheight])
    #hip:
    if len(keypoints)>19:
        hip=0.1
        if all(keypoints[18]) and all(keypoints[19]):
            left_hip_points=np.asarray([round(keypoints[18][2] * w), round(keypoints[18][1] * h)])
            right_hip_points=np.asarray([round(keypoints[19][2] * w), round(keypoints[19][1] * h)])
            hip=pdist(left_hip_points,right_hip_points)
    #waist:
    if len(keypoints)>21:
        waist=0
        if all(keypoints[21]) and all(keypoints[20]):
            left_waist_points=np.asarray([round(keypoints[20][2] * w), round(keypoints[20][1] * h)])
            right_waist_points=np.asarray([round(keypoints[21][2] * w), round(keypoints[21][1] * h)])
            waist=pdist(left_waist_points,right_waist_points)
            waist_height_ratio = round((np.pi*waist/height),4)
            waist_hip_ratio = round((waist/hip),4)
            #print('w:',waist,'hp:',hip,'ht:',height)
            return [waist_height_ratio,waist_hip_ratio]
        else:
            return None

class MyForm(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui=Ui_Form()
        self.ui.setupUi(self)
        self.title='Health Risk Monitoring'
        self.existed_face__key_list = retrive_all_face_keys()
        # #Load the UI Page
        # uic.loadUi('acquire_images_gui.ui', self)
        # self.ui.take_face_picture_btn.clicked.connect(self.take_face_picture)
        # self.ui.retake_face_picture_btn.clicked.connect(self.retake_face_picture)
        # self.ui.take_body_picture_btn.clicked.connect(self.take_body_picture)
        # self.ui.retake_body_picture_btn.clicked.connect(self.retake_body_picture)
        # self.ui.submit_btn.clicked.connect(self.submit_data)
        self.ui.quit_btn.clicked.connect(self.quitf)
        # self.ui.reset_btn.clicked.connect(self.resetb)
        self.face_size=self.ui.original_img.size()
        #self.ui.text_output.setWordWrap(True) 
        #self.ui.history_plot.canvas.axes.axis('off')
        self.modelbmi=load_model(OUTPUT_MODEL_DIR, OUTPUT_MODEL_NAME)
        self.stream_face=True
        self.stream_body=False
        self.image_face=[]
        self.image_body=[]
        self.maindf=pd.DataFrame()
        #yolo:
        self.device = select_device('')
        
        # Load model
        self.model = attempt_load(opt.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(opt.imgsz, s=self.model.stride.max())  # check img_size
        self.model.half()  # to FP16
        #self.img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img

        #self.client = mqtt.Client()
        # self.client.on_connect = self.on_connect
        # self.client.on_disconnect = self.on_disconnect
        # self.client.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
        # self.client.on_message = self.on_message
        # self.client2 = mqtt.Client()
        # self.client2.on_connect = self.on_connect2
        # self.client2.on_disconnect = self.on_disconnect2
        # self.client2.connect(REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, 60)
        # self.client2.on_message = se    display_width=4032,
        #    display_height=3040,f.on_message2
        self.BMI=0        
        self.ssid=str(uuid.uuid4())
        self.ui.label_7.setText(self.ssid)
        #self.ui.history_plot.canvas.axes.axis('on')
        # self.client.loop_start()
        # self.client2.loop_start()
        #self.ui.history_plot.canvas.axes.axis('off')        
        # create a timer
        remove_oneliners()
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.show_img)
        # set control_bt callback clicked  function
        self.controlTimer()
    def get_bmi(self,img):
        # encoding_list=[]
        # print('shape face:',img.shape)
        # face_img = np.asarray(bytearray(img), dtype="uint8")
        # face_img = cv2.imdecode(face_img, cv2.COLOR_BGR2RGB)
        # face_img_encoding = face_recognition.face_encodings(face_img)
        # encoding_list.append(face_img_encoding[0])
        print(self.encoding_list[0].tolist())
        pred_arr = np.expand_dims(np.array(self.encoding_list[0].tolist()), axis=0)
        return np.round(np.exp(self.modelbmi.predict(pred_arr)).item(),4)

    
    def resetb(self):
        print('Starting new session')
        self.BMI=0        
        self.ssid=str(uuid.uuid4())
        print(self.ssid)
        self.ui.label_3.clear()
        self.ui.original_img.clear()
        self.stream_face=True
        self.stream_body=False
        self.image_face=[]
        self.image_body=[]
        # self.ui.history_plot.canvas.axes.cla()
        # self.ui.history_plot.canvas.axes.axis('off')
        # self.ui.history_plot.canvas.axes.plot(np.nan)
        # self.ui.history_plot.canvas.draw()
        self.ui.label_7.setText(self.ssid)
        self.ui.bmi_tag.setText('BMI=')
        self.ui.w2height_label.setText('Waist-to-height ratio:')
        self.ui.w2hip_label.setText('Waist-to-hip ratio:')
        QtWidgets.QApplication.processEvents() 

    def recognit_face(self,new_img):
        print("start finding face")
        #try:
        self.encoding_list = []
        new_face_encoding = face_recognition.face_encodings(new_img)
        for face_img_key in self.existed_face__key_list:
            face_img_object = retrive_data_by_key(face_img_key)
            face_img = np.asarray(bytearray(face_img_object), dtype="uint8")
            face_img = cv2.imdecode(face_img, cv2.COLOR_BGR2RGB)
            face_img_encoding = face_recognition.face_encodings(face_img)
            self.encoding_list.append(face_img_encoding[0])
            print('encoded')
        if len(new_face_encoding)>0:
            result_list = face_recognition.compare_faces(self.encoding_list, new_face_encoding[0])
            find_face = [i for i, x in enumerate(result_list) if x]
            face_id = str(uuid.uuid4())
            # print("result_list ")
            # print(result_list)
            # print("find_face ")
            # print(find_face)
            # if len(find_face) == 0: ## new face
            #     print("new face" + face_id)
            # else:
            if len(find_face) > 0: ## new face
                index = find_face[0]
                face_id = self.existed_face__key_list[index]
                face_id = face_id.split('/')[1].split('.')[0]
                print("existing face " + face_id)
            save_face_data(new_img, face_id)
            #print("face id " + face_id)
        else:
            face_id = str(uuid.uuid4())
            print('NO encoding')
        return face_id
        # except Exception:
        #     print("Error in recognize_face")
        #     traceback.print_exc() 

    def quitf(self):
        #self.timer.stop()
        remove_oneliners()
        print(self.maindf)
        self.controlTimer()
        self.cap.release()
        #cv2.destroyAllWindows()
        self.close()

    def aggregate_object_cloud(self,user_object):
        #face_id = self.recognit_face(user_object[0])
        print("got face id, start updating db")
        #add ID to dataframe
        user_object[1]['ID']=face_id
        #add id to GUI qedit
        self.ui.lineEdit.setText(face_id)
        #Append face and df to the indentified ID and save in S3
        get_and_insert_user_data(user_object[1], face_id)

    def aggregate_object_local(self,user_object):
        #face_id = self.recognit_face(user_object[0])
        #print("got face id, updating db")
        #add ID to dataframe
        #user_object[1]['ID']=face_id
        #add id to GUI qedit
        self.ui.lineEdit.setText(user_object[1]['ID'].values[0])#check if it is zero or -1
        #Append face and df to the indentified ID and save in S3
        self.get_and_insert_user_data_local(user_object[1])

    def get_and_insert_user_data_local(self,update_user_object):
        #update_user_object is a datraframe with ID, timestamp, BMI, w2h and h2w ratios
        #First load exisitng data (last edited csv)
        if len(self.maindf)>0:
            #if np.max(self.maindf.ID.values==update_user_object['ID']):
            self.maindf=self.maindf.append(update_user_object,ignore_index=True)
            #print(self.maindf)
        else:
            self.maindf=update_user_object

    def process_body_results(self,msg):
        self.w2height_ratio=float(msg['waist-height-ratio'])
        self.w2hip_ratio=float(msg['waist-hip-ratio'])
        self.ui.w2height_label.setText('Waist-to-height ratio:'+str(self.w2height_ratio))
        self.ui.w2hip_label.setText('Waist-to-hip ratio:'+str(self.w2hip_ratio))
        self.keypoints=msg['keypoints']
        # for p in msg['keypoints']:
        #     self.image_body=draw_point(p[1],p[0],self.image_body)
        height, width, channel = self.image_body.shape
        ratio=self.ui.label_3.geometry().height()/height
        image = cv2.resize(self.image_body, (int(width*ratio), self.ui.body.geometry().height()))
        # get image infos    display_width=4032,
        #    display_height=3040,image.shape
        step = channel * width
        # create and store QImage from image
        self.qImgb = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.body.setPixmap(QPixmap.fromImage(self.qImgb))
        #self.ui.text_output.setText("Press submit data\n or retake pictures ")

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            #self.cap = cv2.VideoCapture(CAM_INPUT)
            self.cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
            # start timer
            self.timer.start(10)
            # update text
            #self.ui.text_output.setText("press Take Picture")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update text
            #self.ui.text_output.setText("Good bye")
    
    def show_cam(self):
        if self.stream_face:
            self.show_img()
        elif self.stream_body:
            self.show_img()
    
    def get_coordiates(self):
        #self.cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
            #self.cap = cv2.VideoCapture(CAM_INPUT)
        ret, image = self.cap.read()
        #Identify people
        sh=image.shape
        img=np.zeros([1]+list(sh))
        img[0,:,:,:]=image
        # Letterbox
        img= [letterbox(x, new_shape=640, auto=self.rect)[0] for x in img]
        #print('size for Yolo:',img[0].shape)
        # Stack
        img = np.stack(img, 0)
        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
        #Yolov5
        half=True
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred =self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        people=[]
        for det in pred:
            if det is not None and len(det):
                #print(det[:, :4])
                for *xyxy, conf, cls in det:
                        #if names[int(c)]=='person':
                        gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        if cls.detach().cpu().numpy()<0.1:
                            for k in range(len(det)):
                                x,y,w,h=int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])                  
                                height, width,ch= sh
                                fh=height/512
                                fw=width/640
                                crop_img=self.imageo[int(y*fh)-20:int(fh*(y+ h)+20), int(x*fw):int(fw*(x + w))]      
                                people.append(crop_img)
                                #coords=torch.tensor(xywh).view(1, 4).detach().cpu().numpy()[0]            
                            return people #coords,mimg
    def crop_and_show_face(self,frame):
        faces=[]
        #convert to gray scale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray=cv.equalizeHist(gray) #it seems this is not needed
        #detect faces
        self.faceROI=np.zeros([300,300,3])
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            center = (x + w//2, y + h//2)
            self.faceROI = frame[max([0,int(y-0.2*h)]):int(y+h+0.2*h),x:int(x+w+0.2*w)]
            #faces.append(faceROI)
            height, width, channel = self.faceROI.shape
            ratio=self.ui.mugshot.geometry().width()/width
            image = cv2.resize(self.faceROI, (self.ui.mugshot.geometry().width(), int(height*ratio)))
            # get image infos
            height, width, channel = image.shape
            step = channel * width
            # create and store QImage from image
            self.qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
            # show image in img_label
            self.ui.mugshot.setPixmap(QPixmap.fromImage(self.qImg))
            time.sleep(0.1)
    
    def show_img(self):
        ret, image = self.cap.read()
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #store current image objec
        self.imageo=image
        #im-=letterbox(x, new_shape=self.img_size, auto=self.rect)[0]
        #print(image.shape)
        #show camera image
        height, width, channel = image.shape
        ratio=self.ui.original_img.geometry().width()/width
        image = cv2.resize(image, (self.ui.original_img.geometry().width(), int(height*ratio)))
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create and store QImage from image
        self.qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.original_img.setPixmap(QPixmap.fromImage(self.qImg))
        ppl=self.get_coordiates()
        
        self.image_body=np.zeros([300,150,3])
        if not ppl is None:
            #print('Number of identified people: ',len(ppl))
            for i,p in enumerate(ppl):
                #print('Processing subject #',i)
                if not p is None:
                    #show body image()
                    self.image_body=p
                    # height, width, channel = p.shape
                    # ratio=self.ui.body.geometry().height()/height
                    # imb = cv2.resize(p, (self.ui.body.geometry().width(), int(height*ratio)))
                    # # get image infos
                    # height, width, channel = imb.shape
                    # step = channel * width
                    # imb_copy=imb.copy()
                    #estimate_pose
                    
                    #print(keypoints[0])
                    # try:
                    #print('len p:',len(p))
                    keypoints=False
                    ratios=[-1,-1]
                    if len(p)>30:
                        orgimg, keypoints, processed_img = detect_pose(p)
                        if keypoints:
                            ratios  = calculate_ratio(orgimg, keypoints[0])
                            if ratios:
                                if ratios[0]>4:
                                    ratios[0]=-1
                                if ratios[1]>4:
                                    ratios[1]=-1
                                self.ui.w2height_label.setText('Waist-to-height ratio:'+str(ratios[0]))
                                self.ui.w2hip_label.setText('Waist-to-hip ratio:'+str(ratios[1]))

                                self.image_body=np.asarray(processed_img)
                height, width, channel = self.image_body.shape
                if (height>10) and (width>10):
                    ratio=self.ui.body.geometry().height()/height
                    imb = cv2.resize(self.image_body, (self.ui.body.geometry().width(), int(height*ratio)))
                    # get image infos
                    height, width, channel = imb.shape
                    step = channel * width
                    # create and store QImage from image
                    self.qImg = QImage(imb.data, width, height, step, QImage.Format_RGB888)
                    # show image in img_label
                    self.ui.body.setPixmap(QPixmap.fromImage(self.qImg))
                    #Display Mugshot
                    self.crop_and_show_face(p)
                    if np.max(self.faceROI.flatten())>0 and len(self.faceROI.flatten())>32:
                        face_id = self.recognit_face(self.faceROI)
                        self.bmi=self.get_bmi(self.faceROI)
                        self.ui.bmi_tag.setText('BMI:'+str(self.bmi))
                        
                        #make payload as a list with face and df with measurements
                        meas_dict={'ID':face_id,'timestamp':datetime.now(),'w2ht':ratios[0],'w2hp':ratios[1],
                            'BMI':self.bmi,'Location':'Hillview_middleschool','SubLocation':'7-8grades'}
                        dfmetrics=pd.DataFrame([meas_dict])
                        payload=[self.faceROI ,dfmetrics]
                        self.aggregate_object_local(payload)
                        #time.sleep(2)
        else:
            #Display blanks
            ratio=self.ui.body.geometry().height()/height
            imb = cv2.resize(np.zeros(self.image_body.shape), (self.ui.body.geometry().width(), int(height*ratio)))
            # get image infos
            height, width, channel = imb.shape
            step = channel * width
            # create and store QImage from image
            self.qImg = QImage(imb.data, width, height, step, QImage.Format_RGB888)
            # show image in img_label
            self.ui.body.setPixmap(QPixmap.fromImage(self.qImg))



    
    
    
            
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':         
    main()

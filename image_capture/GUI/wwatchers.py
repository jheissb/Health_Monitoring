#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" wwtchers.py opens a GUI to interact with user. 
Face and body pictures are taken and messages are sent to the respective containers for BMI and pose estimation. If detection is OK, data is sent to the cloud for face recognition and historic data is returned. The new data is appended to the database in S3. Program loops every 20 ms cheking if new frames need to be displayed and messages need to be sent"""

from mplwidget import MplWidget
from gui2021 import *
import sys
import cv2
import numpy as np
import threading
import time
import queue
#import paho.mqtt.client as mqtt
import os
import json
import PIL.Image, PIL.ImageDraw
import base64
import uuid
from datetime import datetime
import matplotlib
matplotlib.use('Qt5Agg')
#from pose_detector import detect_pose
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from yolo5.detector import *
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'
#set this to the right camera input
CAM_INPUT=0
#Constants
# LOCAL_MQTT_HOST=os.getenv('local_broker_ip', "172.18.0.2")#"imageprocessorbroker"
# LOCAL_MQTT_PORT=1883
# #Face and Body images
# LOCAL_FACE_MQTT_TOPIC="imagedetection/faceextractor"
# LOCAL_BODY_MQTT_TOPIC="imagedetection/bodyextractor"
# #BMI and ratios
# LOCAL_MQTT_FACE_RESULT_TOPIC="imagedetection/faceprocessor/result"
# LOCAL_MQTT_BODY_RESULT_TOPIC="imagedetection/bodyprocessor/result"

# #Cloud
# REMOTE_MQTT_HOST="44.233.34.126"
# REMOTE_MQTT_PORT=1883
# REMOTE_MQTT_TOPIC="imagedetection/aggregator"
# REMOTE_MQTT_HISTORICAL_DATA="imagedetection/historicaldata"
def draw_point(x,y,img):
    thickness = 5
    height, width, channel = img.shape
    #draw = PIL.ImageDraw.Draw(img)
    #draw.line([ int(x)-1,int(y),int(x)+1,int(y)],width = thickness,fill=(51,51,204))
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
    capture_width=1280*2,
    capture_height=720*2,
    display_width=1280,
    display_height=720,
    framerate=20,
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


def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)


            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

def calculate_ratio(img, keypoints):
  w, h = img.size
  if all(keypoints[1]) and all(keypoints[2]) and all(keypoints[15]) and all(keypoints[16]):
    left_eye_points = np.asarray([round(keypoints[1][2] * w), round(keypoints[1][1] * h)])
    right_eye_points = np.asarray([round(keypoints[2][2] * w), round(keypoints[2][1] * h)])
    left_ankle_points = np.asarray([round(keypoints[15][2] * w), round(keypoints[15][1] * h)])
    right_ankle_points = np.asarray([round(keypoints[16][2] * w), round(keypoints[16][1] * h)])
    left_height = np.linalg.norm(left_ankle_points-left_eye_points)
    right_height = np.linalg.norm(right_ankle_points-right_eye_points)
    avg_height = 1.1*(left_height + right_height) / 2 #add 10% to compensate for eyes and ankles not been at the top and bottom.
    left_hip_points = np.asarray([round(keypoints[11][2] * w), round(keypoints[11][1] * h)])
    right_hip_points = np.asarray([round(keypoints[12][2] * w), round(keypoints[12][1] * h)])  
    left_shoulder_points = np.asarray([round(keypoints[5][2] * w), round(keypoints[5][1] * h)])
    right_shoulder_points = np.asarray([round(keypoints[6][2] * w), round(keypoints[6][1] * h)])
    #estimating waist as midpoint between hip and shoulder
    left_waist_points =(2*left_hip_points+left_shoulder_points)/3
    right_waist_points =(2*right_hip_points+right_shoulder_points)/3
    #avg_shoulder = np.linalg.norm(right_shoulder_points-left_shoulder_points)
    avg_hip = np.linalg.norm(right_hip_points-left_hip_points)
    avg_waist = np.linalg.norm(right_waist_points-left_waist_points)
    waist_height_ratio = round((pi*avg_waist/avg_height),4)
    waist_hip_ratio = round((avg_waist/avg_hip),4)
    return (waist_height_ratio,waist_hip_ratio,list(left_eye_points),list(right_eye_points),list(left_ankle_points),
        list(right_ankle_points),list(left_hip_points),list(right_hip_points),list(left_waist_points),list(right_waist_points))
  else:
    return None


class MyForm(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui=Ui_Form()
        self.ui.setupUi(self)
        self.title='Weight Watchers console'
        # #Load the UI Page
        # uic.loadUi('acquire_images_gui.ui', self)
        # self.ui.take_face_picture_btn.clicked.connect(self.take_face_picture)
        # self.ui.retake_face_picture_btn.clicked.connect(self.retake_face_picture)
        # self.ui.take_body_picture_btn.clicked.connect(self.take_body_picture)
        # self.ui.retake_body_picture_btn.clicked.connect(self.retake_body_picture)
        # self.ui.submit_btn.clicked.connect(self.submit_data)
        # self.ui.quit_btn.clicked.connect(self.quitf)
        # self.ui.reset_btn.clicked.connect(self.resetb)
        self.face_size=self.ui.face.size()
        self.ui.text_output.setWordWrap(True) 
        #self.ui.history_plot.canvas.axes.axis('off')

        self.stream_face=True
        self.stream_body=False
        self.image_face=[]
        self.image_body=[]
        #self.client = mqtt.Client()
        # self.client.on_connect = self.on_connect
        # self.client.on_disconnect = self.on_disconnect
        # self.client.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
        # self.client.on_message = self.on_message
        # self.client2 = mqtt.Client()
        # self.client2.on_connect = self.on_connect2
        # self.client2.on_disconnect = self.on_disconnect2
        # self.client2.connect(REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, 60)
        # self.client2.on_message = self.on_message2
        self.BMI=0        
        self.ssid=str(uuid.uuid4())
        self.ui.label_7.setText(self.ssid)
        #self.ui.history_plot.canvas.axes.axis('on')
        # self.client.loop_start()
        # self.client2.loop_start()
        #self.ui.history_plot.canvas.axes.axis('off')        
        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.show_cam)
        # set control_bt callback clicked  function
        self.controlTimer()

    def resetb(self):
        print('Starting new session')
        self.BMI=0        
        self.ssid=str(uuid.uuid4())
        print(self.ssid)
        self.ui.label_3.clear()
        self.ui.face.clear()
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

    def quitf(self):
        self.client.loop_stop()
        #self.timer.stop()
        self.controlTimer()
        self.cap.release()
        cv2.destroyAllWindows()
        self.close()

    def on_connect2(self,client, userdata, flags, rc):
        #subscribe to topics 
        print("Connected to remote server with result code "+str(rc))
        self.client2.subscribe(REMOTE_MQTT_HISTORICAL_DATA)

    def on_connect(self,client, userdata, flags, rc):
        #subscribe to topics 
        print("Connected with result code "+str(rc))
        self.client.subscribe(LOCAL_MQTT_FACE_RESULT_TOPIC)
        self.client.subscribe(LOCAL_MQTT_BODY_RESULT_TOPIC)
        #self.client.subscribe(REMOTE_MQTT_HITORICAL_DATA)

    def on_disconnect(self,client, userdata,rc=0):
        self.client.loop_stop()
    def on_disconnect2(self,client, userdata,rc=0):
        self.client2.loop_stop()
    def publish_face(self,payload):
        self.client.publish(LOCAL_FACE_MQTT_TOPIC, payload, qos=1, retain=False)
        print('face published')
    def publish_body(self,payload):
        self.client.publish(LOCAL_BODY_MQTT_TOPIC, payload, qos=1, retain=False)

    def on_message(self,client,userdata, msg):
        #if msg.topic == REMOTE_MQTT_HITORICAL_DATA: 
        #    self.process_historical_data(msg.payload)
        if msg.topic == LOCAL_MQTT_FACE_RESULT_TOPIC: 
            BMI=msg.payload.decode("utf-8") #.decode("ascii")
            BMI=BMI[0:5]            
            print('bmi message=',BMI)
            self.ui.bmi_tag.setText('BMI= '+BMI)
            self.BMI=float(BMI)
        if msg.topic == LOCAL_MQTT_BODY_RESULT_TOPIC: 
            #try:
            m_decode=str(msg.payload.decode("utf-8"))
            pose=json.loads(m_decode)
            print(pose)
            self.process_body_results(pose)
            #except Exception as e:
            #    print("error in on_message")
            #    print(str(e))                    
    def on_message2(self,client,userdata, msg):
        print('Got a message from cloud server')
        if msg.topic == REMOTE_MQTT_HISTORICAL_DATA: 
            #try:
            print('Reading historical data...')
            m_decode=str(msg.payload.decode("utf-8"))
            hdata=json.loads(m_decode)
            #reading list of dictionaries with date, bmi, w2height ratio and w2hip ratio
            list_d = hdata['history']
            self.ssid=hdata['session-id']
            dates_l=[]
            bmi_l=[]
            w2height_l=[]
            w2hip_l=[]
            for d in list_d:
                dates_l.append(d['date'].rsplit('-',1)[0]) #removing seconds
                bmi_l.append(d['bmi'])
                w2height_l.append(d['waist-height-ratio'])
                w2hip_l.append(d['waist-hip-ratio'])
            #self.ui.history_plot.canvas.axes.axis('on')
            #ax2=self.ui.history_plot.canvas.axes.twinx()
            # self.ui.history_plot.canvas.axes.plot(dates_l,bmi_l,'o-',label='BMI')         
            # self.ui.history_plot.canvas.axes.plot(dates_l,10*np.array(w2height_l),'o-',label='10 x W2Height')
            # self.ui.history_plot.canvas.axes.plot(dates_l,10*np.array(w2hip_l),'o-',label='10 x W2Hip')
            # self.ui.history_plot.canvas.axes.set_xticklabels(dates_l,rotation=45)
            # self.ui.history_plot.canvas.axes.legend()
            # self.ui.history_plot.canvas.draw()
            # self.ui.text_output.setText('Press Reset to continue or Quit')
       
    def process_body_results(self,msg):
        self.w2height_ratio=float(msg['waist-height-ratio'])
        self.w2hip_ratio=float(msg['waist-hip-ratio'])
        self.ui.w2height_label.setText('Waist-to-height ratio:'+str(self.w2height_ratio))
        self.ui.w2hip_label.setText('Waist-to-hip ratio:'+str(self.w2hip_ratio))
        self.keypoints=msg['keypoints']
        for p in msg['keypoints']:
            self.image_body=draw_point(p[1],p[0],self.image_body)
        height, width, channel = self.image_body.shape
        ratio=self.ui.label_3.geometry().height()/height
        image = cv2.resize(self.image_body, (int(width*ratio), self.ui.face.geometry().height()))
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create and store QImage from image
        self.qImgb = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.label_3.setPixmap(QPixmap.fromImage(self.qImgb))
        self.ui.text_output.setText("Press submit data\n or retake pictures ")

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
            #self.cap = cv2.VideoCapture(CAM_INPUT)
            ret, image = self.cap.read()
            # start timer
            self.timer.start(20)
            # update text
            self.ui.text_output.setText("press Take Picture")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update text
            self.ui.text_output.setText("Good bye")
    def show_cam(self):
        if self.stream_face:
            self.show_face_cam()
        elif self.stream_body:
            self.show_body_cam()
    
    def take_body_picture(self):
        if self.ui.delay.isChecked():
            for i in range(70):
                self.show_body_cam()
                time.sleep(0.01)
                QtWidgets.QApplication.processEvents() 
        self.image_body=self.imagebo
       
        # except:
        #     print('Retake body image')



        
        #self.ui.text_output.setText('Sending body image')
        self.image_face=self.imageo
        #send message to ratio estimator and read incoming ratios
        # rc,png = cv2.imencode('.png', self.image_body)
        # msg = png.tostring()
        # self.publish_body(msg)
        # print("Sent detected body to mosquitto")
        # self.client.on_message = self.on_message
        # self.stream_body=False
 

    def retake_body_picture(self):
        self.stream_body=True
        self.image_body=[]
        self.stream_face=False
        self.ui.text_output.setText("Take body picture")

    #def on_message(self,client,userdata, msg):
    #    self.BMI=msg

    def submit_data(self):
        _,img_face = cv2.imencode('.png', self.image_face)
        png_face_as_text = base64.b64encode(img_face).decode()
        _,img_body = cv2.imencode('.png', self.image_body)
        png_body_as_text = base64.b64encode(img_body).decode()
        user_object = {}
        user_object['face-img'] = png_face_as_text
        user_object['bmi'] = self.BMI
        user_object['waist-height-ratio'] = self.w2height_ratio
        user_object['waist-hip-ratio'] = self.w2hip_ratio
        user_object['keypoints'] = self.keypoints
        user_object['body-img'] = png_body_as_text
        user_object['session-id'] =self.ssid
        user_object['timestamp']=datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        payload=json.dumps(user_object, ensure_ascii=False, indent=4)
        self.client2.publish(REMOTE_MQTT_TOPIC, payload, qos=0, retain=False)
        self.ui.text_output.setText("Data sent to cloud.\n Receiving history")

    def take_face_picture(self):
        """takes picture of face from webcam:
        fixes the frame in current and stop streaming on the face canvas"""
        self.stream_face=False
        if len(self.image_body)<1:
            self.stream_body=True
        self.ui.text_output.setText('body:'+str(self.stream_body))
        self.image_face=self.imageo
        #send message to BMI estimator and read BMI
        rc,png = cv2.imencode('.png', self.image_face)
        msg = png.tostring()
        self.publish_face(msg)
        print("Sent detected face to mosquitto")
        #self.client.on_message = self.on_message

    def retake_face_picture(self):
        self.stream_face=True
        self.image_face=None
        self.ui.text_output.setText("Take face picture")
        self.show_face_cam()

    def show_face_cam(self):
        ret, image = self.cap.read()
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #store current image object
        self.imageo=image
        height, width, channel = image.shape
        ratio=self.ui.face.geometry().width()/width
        image = cv2.resize(image, (self.ui.face.geometry().width(), int(height*ratio)))
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create and store QImage from image
        self.qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.face.setPixmap(QPixmap.fromImage(self.qImg))
            
            
    def show_body_cam(self):
        ret, image = self.cap.read()
        # convert image to RGB format
        imagec = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #find pose
        # orgimg, keypoints, processed_img = detect_pose(imagec)
        # try:
        #     if keypoints:
        #         ratios_and_points  = calculate_ratio(orgimg, keypoints[0])
        #         body_image = {}
        #         #print('calculations:',ratios_and_points)
        #         #_,png = cv2.imencode('.png', processed_img)
        #         body_image['waist-hip-ratio'] = ratios_and_points[1]
        #         body_image['waist-height-ratio'] = ratios_and_points[0]
        #         body_image['keypoints'] = ratios_and_points[2:] 
        #         for p in body_image['keypoints']:
        #             self.image_body=draw_point(p[1],p[0],imagec)
        #         #store current image object
        #         imagec=self.image_body
        # except:
        #     None
        height, width, channel = imagec.shape
        ratio=self.ui.label_3.geometry().height()/height
        image = cv2.resize(imagec, (int(width*ratio), self.ui.face.geometry().height()))
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create and store QImage from image
        self.qImgb = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.label_3.setPixmap(QPixmap.fromImage(self.qImgb))
        #self.ui.text_output.setText("Take face and body picture then submit data ")
            
            
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MyForm()
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':         
    main()

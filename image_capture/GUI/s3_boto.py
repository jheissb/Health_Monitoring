#THE CURD method for s3
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import uuid
import os
import io
from PIL import Image as Image
from array import array
import json
import sys
import logging
import traceback
import base64
import cv2
from collections.abc import Iterable # for python >= 3.6
import pandas as pd

# logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#credentials = json.load(open('aws_cred.json'))

session = boto3.Session(
         aws_access_key_id='AKIA6AFBQIPHSC5WGJEN',
         aws_secret_access_key='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')
  
s3_client = session.client('s3')



S3_BUCKET_NAME='weightwatcher'
S3_USER_HISTORICAL_FOLDER_NAME='userdata'
S3_USER_DATA_KEY_FORMAT=S3_USER_HISTORICAL_FOLDER_NAME+"/{id}.csv"
S3_FACE_ID_FOLDER_NAME='faceid'
S3_FACE_ID_KEY_FORMAT=S3_FACE_ID_FOLDER_NAME+"/{id}.png"

def save_face_data(face_img, face_id):
    try:
        #face_img=cv2.imdecode(face_img, cv2.COLOR_BGR2RGB)
        _,png = cv2.imencode('.png', face_img[..., ::-1])
        img = io.BytesIO(png.tobytes())
        img.seek(0)
        key = S3_FACE_ID_KEY_FORMAT.format(id=face_id)
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=key, Body=img)
    except Exception: 
        print('Error saving img in S3')
        logger.error("Error on save_face_data")
        traceback.print_exc() 

def remove_oneliners():
    #bk = boto3.conn.get_bucket('my_bucket_name')
    prefix = S3_USER_HISTORICAL_FOLDER_NAME
    Response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
    Files_ListS = Response.get('Contents')
    if isinstance(Files_ListS, Iterable):
        for f in Files_ListS:
            if f['Key'].endswith('.csv'):
                if f['Size']<200:
                    print('deleting ',f['Key'])
                    s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=f['Key'])
                    


def get_and_insert_user_data(update_user_object, face_id):
    #update_user_object is a datraframe with ID, timestamp, BMI, w2h and h2w ratios
    try:
        #First load exisitng data (last edited csv)
        prefix = "{}/{}.csv".format(S3_USER_HISTORICAL_FOLDER_NAME, face_id)
        Response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
        Files_ListS = Response.get('Contents')
        if isinstance(Files_ListS, Iterable):
            csv_objects = [f for f in Files_ListS if f['Key'].endswith('.csv')]
            if len(csv_objects)>0:
                #historical_data_fn=max(csv_objects, key=lambda x: x['LastModified'])
                obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=prefix)
                initial_df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        else:
            initial_df=pd.DataFrame()
        #Then append current CSV to exisitng data and upload
        update_user_object=initial_df.append(update_user_object)
        s3_key = S3_USER_DATA_KEY_FORMAT.format(id=face_id)
        csv_msg_df=update_user_object#pd.DataFrame([update_user_object])
        #print(update_user_object)
        csv_buf = io.StringIO()
        csv_msg_df.to_csv(csv_buf, header=True, index=False)
        csv_buf.seek(0)
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Body=csv_buf.getvalue(), Key=s3_key)

    except Exception:
        logger.error("get_and_insert_user_data")
        traceback.print_exc()

def retrive_all_face_keys():
    try:
        # prefix = "{}/{}".format(S3_BUCKET_NAME, S3_FACE_ID_FOLDER_NAME)
        objects = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME,Prefix=S3_FACE_ID_FOLDER_NAME)

        user_historical_data = []
        for key_object in objects['Contents']:
            user_historical_data.append(key_object['Key'])
        print(user_historical_data)
        return user_historical_data[1:]
    except Exception:
        logger.error("retrive_all_face_keys")
        traceback.print_exc()

# #TODO: deal with truncated file
# def retrive_user_hitstorical_data_by_face_id(face_id):
#     try:
#         prefix = "{}/{}".format(S3_USER_HISTORICAL_FOLDER_NAME, face_id)
#         objects = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME,Prefix=prefix)

#         user_historical_data = []
#         for key_object in objects['Contents']:
#             key_split = key_object['Key'].split("/")
#             historical_object = {}
#             historical_data = retrive_data_by_key(key_object['Key']).decode()
#             historical_data = json.loads(historical_data)
#             user_file_name = key_split[-1]
#             historical_object['date']=user_file_name.split(".")[0]
#             historical_object['bmi']=historical_data['bmi'] if historical_data['bmi'] else 0
#             historical_object['waist-height-ratio']= historical_data['waist-height-ratio'] if historical_data['waist-height-ratio'] else 0
#             historical_object['waist-hip-ratio']=historical_data['waist-hip-ratio'] if historical_data['waist-hip-ratio'] else 0
#             user_historical_data.append(historical_object)
#         return user_historical_data
#     except Exception:
#         logger.error("retrive_user_hitstorical_data_by_face_id")
#         traceback.print_exc()

def retrive_data_by_key(key):
    try:
        s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key) 
        return s3_object["Body"].read()
    except Exception:
        logger.error("retrive_user_historical_data_by_date_and_face_id")
        traceback.print_exc()

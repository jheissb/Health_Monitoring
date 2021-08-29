#!/usr/bin/env python
# coding: utf-8

# ### Make Predictions on new images

# In[40]:


from keras.models import model_from_json
import cv2
import dlib
from skimage.transform import resize
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
import pandas as pd


# In[41]:


### HERE comes the image collection code


# In[42]:


## Test on Valid dataset


# In[43]:


dataset_val=pd.read_csv('../data/valid.csv')
dataset_val.head()


# In[44]:


dataset_val.shape


# In[45]:


dataset_val=dataset_val[dataset_val['bmi']<=40]


# In[46]:


all_paths_val=dataset_val['index'].tolist()
all_bmi_val=dataset_val['bmi']


# In[ ]:





# In[47]:


def preprocess_image(img=None,data=None,image_path=None):
    if image_path:
        image_path=image_path#'data/face/'+str(path)
        img = cv2.imread(image_path)
        data= image.imread(image_path)
    
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)
    if len(faces)>0:
        x1 = faces[0].left() # left point
        y1 = faces[0].top() # top point
        x2 = faces[0].right() # right point
        y2 = faces[0].bottom()
        face_out=cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0))

        ## Add some margin on the faces
        margin20_x=int(((x2-x1)*0.2)//2)
        margin20_y=int(((y2-y1)*0.2)//2)

        margin10_x=int(((x2-x1)*0.1)//2)
        margin10_y=int(((y2-y1)*0.1)//2)

        output= data[y1-margin20_y:y2+margin20_y,x1-margin20_x:x2+margin20_x,:]

        #resized_data = resize(output, (300, 300, 3))

        resized_data = cv2.resize(output, (300, 300), interpolation = cv2.INTER_AREA)
    else:
        resized_data=None
        
    return resized_data


# In[48]:


new_image=preprocess_image(image_path='../data/test/single_face/emma_watson.jpg')


# In[49]:


plt.imshow(new_image)


# In[50]:


import pickle
# model_without_tl_without_aug
# model_without_tl_with_aug
# model_with_tl_without_aug
# model_with_tl_with_aug

model_name1='model_without_tl_without_aug'
json_file = open(model_name1+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model1 = model_from_json(loaded_model_json)
# load weights into new model
loaded_model1.load_weights(model_name1+".h5")



model_name2='model_without_tl_with_aug'
json_file = open(model_name2+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model2 = model_from_json(loaded_model_json)
# load weights into new model
loaded_model2.load_weights(model_name2+".h5")



model_name3='model_with_tl_without_aug'
json_file = open(model_name3+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model3 = model_from_json(loaded_model_json)
# load weights into new model
loaded_model3.load_weights(model_name3+".h5")


model_name4='model_with_tl_with_aug'
json_file = open(model_name4+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model4 = model_from_json(loaded_model_json)
# load weights into new model
loaded_model4.load_weights(model_name4+".h5")



model_name6='model_random_forest.pkl'
with open(model_name6,'rb') as file:
    loaded_model6=pickle.load(file)
# y_pred_rf=model_loaded.predict(X_test)
# y_pred_train_rf=model_loaded.predict(X_train)


model_name7='model_with_tl_without_aug_moredata'
json_file = open(model_name7+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model7 = model_from_json(loaded_model_json)
# load weights into new model
loaded_model7.load_weights(model_name7+".h5")

print("Loaded models from disk")


# In[51]:


#!pip install face_recognition


# In[52]:


import face_recognition

def get_face_encoding_from_message(msg):
    my_face_encoding = face_recognition.face_encodings(msg)
    if not my_face_encoding:
        return np.zeros(128).tolist()
    return my_face_encoding[0].tolist()


# In[53]:


new_image.shape


# In[54]:


true_bmi=[]
predicted_bmi1=[]
predicted_bmi2=[]
predicted_bmi3=[]
predicted_bmi4=[]

predicted_bmi6=[]
predicted_bmi7=[]

for ii,path in enumerate(all_paths_val[:]):
    image_path='../data/face/'+str(path)
    
    img = cv2.imread(image_path)
    data = image.imread(image_path)
    new_image=preprocess_image(img,data,image_path=None)
    if isinstance(new_image,np.ndarray):
        new_image1=np.expand_dims(new_image,axis=0)
        #plt.imshow(data)
        #plt.show()
        plt.imshow(new_image1[0,:,:,:])
        plt.show()
        bmi_predict=loaded_model1.predict(new_image1/255.0)
        bmi_predict2=loaded_model2.predict(new_image1/255.0)
        bmi_predict3=loaded_model3.predict(new_image1/255.0)
        bmi_predict4=loaded_model4.predict(new_image1/255.0)
        
        bmi_predict7=loaded_model7.predict(new_image1/255.0)

        
        
        bmi_predict6=loaded_model6.predict(np.array(get_face_encoding_from_message(new_image)).reshape(1,-1))
        
        print('True BMI', all_bmi_val.iloc[ii])
        print('Predicted BMI 1: ',bmi_predict[0][0])
        print('Predicted BMI 2: ',bmi_predict2[0][0])
        print('Predicted BMI 3: ',bmi_predict3[0][0])
        print('Predicted BMI 4: ',bmi_predict4[0][0])
        
        print('Predicted BMI 6: ',bmi_predict6[0])
        print('Predicted BMI 7: ',bmi_predict7[0][0])
#         print('Difference: ',np.abs(all_bmi_val.iloc[ii]-bmi_predict3[0][0]))
        print('*************************************************************************\n')
        true_bmi.append(all_bmi_val.iloc[ii])
        predicted_bmi1.append(bmi_predict[0][0])
        predicted_bmi2.append(bmi_predict2[0][0])
        predicted_bmi3.append(bmi_predict3[0][0])
        predicted_bmi4.append(bmi_predict4[0][0])
        
        predicted_bmi7.append(bmi_predict7[0][0])
        
        predicted_bmi6.append(bmi_predict6[0])


# In[55]:


plt.figure(figsize=(25,8))
plt.plot(true_bmi,'x-',label='True BMI')
plt.plot(predicted_bmi3,'o-',label='Predicted BMI3')
plt.legend()
plt.xlabel('Samples')
plt.ylabel('BMI')


# In[56]:


plt.figure(figsize=(25,8))
plt.plot(true_bmi,'x-',label='True BMI')
plt.plot(predicted_bmi6,'o-',label='Predicted BMI RF')
plt.legend()
plt.xlabel('Samples')
plt.ylabel('BMI')


# In[57]:


truenp=np.array(true_bmi)
predictnp=np.array(predicted_bmi3)


# In[58]:


plt.figure(figsize=(30,6))
plt.plot(truenp[truenp<35],'x-',label='True BMI')
# plt.plot(predicted_bmi1,'o-',label='Predicted BMI1')
# plt.plot(predicted_bmi2,'o-',label='Predicted BMI2')
plt.plot(predictnp[truenp<35],'o-',label='Predicted BMI3')
#plt.plot(predicted_bmi4,'o-',label='Predicted BMI4')
plt.legend()
plt.xlabel('Samples')
plt.ylabel('BMI')


# In[59]:


plt.figure(figsize=(30,6))
plt.plot(true_bmi[:50],'x-',label='True BMI')
# plt.plot(predicted_bmi1,'o-',label='Predicted BMI1')
# plt.plot(predicted_bmi2,'o-',label='Predicted BMI2')
plt.plot(predicted_bmi3[:50],'o-',label='Predicted BMI3')
#plt.plot(predicted_bmi4,'o-',label='Predicted BMI4')
plt.legend()
plt.xlabel('Samples')
plt.ylabel('BMI')


# In[60]:


plt.figure(figsize=(30,7))
plt.scatter(np.arange(0,50,1),true_bmi[:50],label='True BMI')
# plt.plot(predicted_bmi1,'o-',label='Predicted BMI1')
# plt.plot(predicted_bmi2,'o-',label='Predicted BMI2')
plt.scatter(np.arange(0,50,1),predicted_bmi3[:50],label='Predicted BMI3',marker='x')
#plt.plot(predicted_bmi4,'o-',label='Predicted BMI4')
plt.legend()
plt.xlabel('Samples')
plt.ylabel('BMI')


# In[61]:


truenp=np.array(true_bmi)
predictnp3=np.array(predicted_bmi3)
predictnp7=np.array(predicted_bmi7)


# In[62]:


plt.figure(figsize=(23,12))

plt.xlabel('True BMI',fontsize=18)
plt.ylabel('Predicted BMI CNN', fontsize=18)


#create basic scatterplot
plt.plot(true_bmi,predicted_bmi3, 'o')
#plt.plot(truenp[truenp<35],predictnp3[predictnp3<34], 'o')

plt.plot([15,40],[15,40],'g--')


# In[73]:


plt.figure(figsize=(23,12))

plt.xlabel('True BMI', fontsize=18)
plt.ylabel('Predicted BMI CNN (More Data)', fontsize=18)


#create basic scatterplot
plt.plot(true_bmi,predicted_bmi7, 'o')
#plt.plot(truenp[truenp<37],predictnp7[predictnp7<31], 'o')

plt.plot([15,40],[15,40],'g--')


# In[64]:


plt.figure(figsize=(23,12))

plt.xlabel('True BMI', fontsize=18)
plt.ylabel('Predicted BMI RF', fontsize=18)


#create basic scatterplot
plt.plot(true_bmi,predicted_bmi6, 'o')

plt.plot([15,40],[15,40],'g--')


# In[65]:


## Mean Absolute Error Model 1
np.abs(np.array(true_bmi) - np.array(predicted_bmi1)).mean()


# In[27]:


## Mean Absolute Error Model 2
np.abs(np.array(true_bmi) - np.array(predicted_bmi2)).mean()


# In[28]:


## Mean Absolute Error Model 3
np.abs(np.array(true_bmi) - np.array(predicted_bmi3)).mean()


# In[29]:


## Mean Absolute Error Model 4
np.abs(np.array(true_bmi) - np.array(predicted_bmi4)).mean()


# In[30]:


## Mean Absolute Error Model 6
np.abs(np.array(true_bmi) - np.array(predicted_bmi6)).mean()


# In[66]:


## Mean Absolute Error Model 7
np.abs(np.array(true_bmi) - np.array(predicted_bmi7)).mean()


# ### Histograms for BMI

# In[67]:


plt.figure(figsize=(10,5))
plt.hist(true_bmi,bins=50)
plt.xlim(15,40)
plt.xlabel('BMI')
plt.ylabel('Count')
plt.title('Validation Dataset: True BMI')
plt.show()


plt.figure(figsize=(10,5))
plt.hist(predicted_bmi1,bins=50)
plt.xlim(15,40)
plt.xlabel('BMI')
plt.ylabel('Count')
plt.title('Validation Dataset: Predicted BMI, model 1')
plt.show()


plt.figure(figsize=(10,5))
plt.hist(predicted_bmi2,bins=50)
plt.xlim(15,40)
plt.xlabel('BMI')
plt.ylabel('Count')
plt.title('Validation Dataset: Predicted BMI, model 2')
plt.show()

plt.figure(figsize=(10,5))
plt.hist(predicted_bmi3,bins=50)
plt.xlim(15,40)
plt.xlabel('BMI')
plt.ylabel('Count')
plt.title('Validation Dataset: Predicted BMI, model 3')
plt.show()


plt.figure(figsize=(10,5))
plt.hist(predicted_bmi4,bins=50)
plt.xlim(15,40)
plt.xlabel('BMI')
plt.ylabel('Count')
plt.title('Validation Dataset: Predicted BMI, model 4')
plt.show()

plt.figure(figsize=(10,5))
plt.hist(predicted_bmi6,bins=50)
plt.xlim(15,40)
plt.xlabel('BMI')
plt.ylabel('Count')
plt.title('Validation Dataset: Predicted BMI, model 6')
plt.show()

plt.figure(figsize=(10,5))
plt.hist(predicted_bmi7,bins=50)
plt.xlim(15,40)
plt.xlabel('BMI')
plt.ylabel('Count')
plt.title('Validation Dataset: Predicted BMI, model 7')
plt.show()


# In[ ]:





# In[ ]:





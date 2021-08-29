import face_recognition
import csv
from PIL import Image
import os
import glob
import numpy as np
import config
from utility import load_model
from model import predict_bmi
import json
import datetime
import random




def face_main( input_path ):
    print("loading trained BMI model...")
    model = load_model(config.OUTPUT_MODEL_DIR, config.OUTPUT_MODEL_NAME)
    encoding_list = []
    num_files = 0
    bmi_list = []
    ID_each_pic = []
    bmi_each_pic = []

    data_length = 0;
    for image_file in os.listdir( input_path ):
        new_img = face_recognition.load_image_file( os.path.join( input_path, image_file ) )
        new_face_encoding = face_recognition.face_encodings( new_img )
        if new_face_encoding:  ## if is a face
            ## only check the first one
            face_encoding = new_face_encoding[0]
            result_list = face_recognition.compare_faces(encoding_list, face_encoding )
            face = face_encoding.tolist();
            
            if len([i for i, x in enumerate(result_list) if x]) == 0: ## new face   
                encoding_list.append( face ) 
                bmi = face_predict( face, model )   
                bmi_list.append([bmi])
                data_length = data_length + 1
                #data_base[face_encoding] = [data_length, bmi]
                ID = data_length - 1
                print(image_file, ": new face, ID = ", ID, " BMI : %2.1f" % (bmi))
            else: ## old face 
                old_face_idx = next((i for i, x in enumerate(result_list) if x), None);
                old_bmi = bmi_list[old_face_idx]
                bmi = face_predict( face, model )
                old_bmi.append( bmi )
                bmi_list[old_face_idx] = old_bmi
                ID = old_face_idx
                print(image_file, ": old face, ID = ", ID, "\t BMI :", ' '.join(['{:2.1f}'.format(x) for x in old_bmi])) 
            
            ID_each_pic.append(ID)
            bmi_each_pic.append(bmi)

    # current date and time
    timestamp = datetime.datetime.now()

    camera_id = random.randint(1,5)
    location_list = ['Bethany','Altamont','Questa']
    sublocation_list = ['Grade-1','Grade-2','Grade-3']


## write to a file
    with open ('data_face2bmi_rong.csv', 'w') as jf:
        thewriter = csv.writer(jf)
        thewriter.writerow(['record','id','timestamp','image','camera_id','location','sublocation','bmi','wt2heightration','wst2hipratio'])
        for i in range(len(ID_each_pic)):
            thewriter.writerow([i+1,ID_each_pic[i],timestamp,'1',random.randint(1,5),random.choice(location_list),random.choice(sublocation_list),bmi_each_pic[i],random.uniform(0.3,0.6),random.uniform(0.6,1.0)])
        #l1 = json.dumps(encoding_list)
        #l2 = json.dumps(bmi_list)
        #jf.write(l1)
        #jf.write('\n')
        #jf.write(l2)


def face_predict( face_list, model ):
    pred_arr = np.expand_dims(np.array(face_list), axis=0)
    return (np.exp(model.predict(pred_arr))).item()


face_main("../rong_pic")

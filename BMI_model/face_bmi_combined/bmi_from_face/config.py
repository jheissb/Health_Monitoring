# training config
#from keras.models import model_from_json

IMGS_DIR = 'images'
IMGS_INFO_FILE = 'data/data_schema_demo.csv'

OUTPUT_MODEL_DIR = 'models'
OUTPUT_MODEL_NAME = 'bmi_RF.model'
#model_name3='model_with_tl_without_aug'
#json_file = open(OUTPUT_MODEL_DIR + '/' + model_name3+'.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model3 = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model3.load_weights(model_name3+".h5")

OUTPUT_TRAINING_DATA_DIR = 'data'
OUTPUT_TRAINING_DATA_FILE = 'train_full_df.csv'

# prediction config
#PREDICTION_IMG_PATH = 'prediction_images/leonardo_d.jpg'
PREDICTION_IMG_PATH = '../rong_pic'

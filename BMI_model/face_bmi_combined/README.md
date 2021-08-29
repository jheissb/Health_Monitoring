input: rong-pic folder

output: data_face2bmi_rong.csv file, 
which includes data for future vizuliaztion purpose

model used: bmi_FR.model from the models folder

aws access key is within secrets.py file 

step 1: in your terminal: run python3 face2bmi.py  --get the data_face2bmi.txt file generated


step 2: in your terminal: run automatic_s3_uploader.py -- send the data_face2bmi.txt file to s3






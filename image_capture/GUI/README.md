# GUI to demonstrate the program: wwatchers2021.py

The program will detect people, crop a body image, identify anaotmical landmarks, use those landmarks to compute height, waist and hip, and calculate waist-to-height and waist to hip ratios. Then the face is segmented and used to estimate BMI using the random forest model. When pressing quit, the data is uploaded to the cloud. This will be replaced by robust mean of each subject and then upload to cloud at the end of each day, by appending to an exisitng master CSV file in S3.


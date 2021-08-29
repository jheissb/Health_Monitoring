# BMI Model Details

This folder contains the details of our BMI training that includes Datasets, Pythion code for different Models and final models.

Predicting BMI is not an easy task. 

We initially started with around 1250 records for our training and later on added another 1050 images. 

We tried several different approaches.

## Dataset:
 - Initial set of 1253 training records and 282 validation records
 - Added 1050 additional images to the dataset

## Approaches:

- Used Random Forest for training the model 
- Used custom Keras deep learning network for training
- Also used transfer learning (1) starting with ResNet50 / VGG19 model (20 million parameters)
- Also used transfer learning (2) starting with VGGFace model (36 million trainable parameters) 
- Used data augmentation - did not see any improvement in prediction
- Also used categories instead of absolute BMI

## Tracking Trends (Our journey through an example)

![image](https://user-images.githubusercontent.com/64849289/128396142-4c8358a8-e40f-4a54-846c-f3ccfe0a59be.png)

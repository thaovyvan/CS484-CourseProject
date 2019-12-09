# CNN stuff
#from keras.models import Sequential
#from keras.layers import Convolution2D
import math
from scipy.misc import imresize, imread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from skimage import filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value

#Possible helper method that can link each image to its class
#Training model (for now)
def classes_and_images():
    return None

#Set up our CNN training model
def training_model():
    return None

#Evaluate the results of our training model by running some predictions
def evaluation():
    return None

#Show the results of our predictions via confusion matrix
def confusion():
    return None
#Takes the image we loaded and standardize scaling, dimensions, etc.
#Preprocessing for training set (for now)
def standardize(image, min):
    new_image = imread('./food41/images/'+ image + '.jpg')
    try:
        width, height, z = image.shape
        if (width < min):
            x = int((float(height)*float(min/float(width))))
            new_image2 = imresize(new_image, (min, x))
        elif(height< min):
            yy = int((float(width)*float(min/float(height))))
            new_image2 = imresize(new_image, (min, y))

    except:
        print("Image error. Skipping...")

    gen = ImageDataGenerator(zoom_range = [.9,1],
    horizontal_flip=false,
    vertical_flip=false,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization = False,
    samplewise_std_normalization=False,
    channel_shift_range=0,
    rotation_range = 0,
    fill_mode="reflect")
    
    gen.config["random_crop_size"] = (min, min)
    trainingModel = gen
    return new_image2

def main():
    resized_array = []
    class_array = []
    trainData = open('./food41/meta/meta/train.txt', 'r')
    train_foods = trainData.read().splitlines()
    for line in train_foods:
        line2 = line.split("/")
        class.append(line2[0])
    print(train_foods)
    print("HELLOLOLOL0 CHIKCHENS")
    for line in train_foods:
        image = Image.open('./food41/images/'+ line + '.jpg', 'r')
        new_image = standarize(image, 300)
        resized_array.add(new_image)
    #img = toMatrix(train_foods[0])
    #classifier = Sequential()

main()

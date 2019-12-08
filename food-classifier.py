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

#This is all of our images we train off of
#Open them and make it pretty
trainData = open('./food41/meta/meta/train.txt', 'r')
train_foods = trainData.read().splitlines()

#Takes the image and turns it into multidimensional list
#that is based off of the image's width and height
    '''
def toMatrix(location):
    image = Image.open('./food41/images/'+location + '.jpg', 'r')
    width, height = image.size
    pix_matrix = []
    pix_val = list(image.getdata())
    for x in range(0, height):
        temp = []
        for y in range(x, x + width):
            temp.append(pix_val[y])
        pix_matrix.append(temp)
    return pix_matrix
    '''

#Takes the image we loaded and standardize scaling, dimensions, etc.
#Preprocessing for training set (for now)
def standardize(image, min):
    new_image = imread('./food41/images/'+ image + '.jpg')
    try:
        width, height, z = image.shape
        if (width < min):
            x = int((float(height)*float(min/float(width))))
            new_image = imresize(new_image, (min, x))
        elif(height< min):
            yy = int((float(width)*float(min/float(height))))
            new_image = imresize(new_image, (min, y))

    except:
        print("Image error. Skipping...")
    return new_image

def main():
    resized_array = []
    print(train_foods)
    print("HELLOLOLOL0 CHIKCHENS")
    for line in train_foods:
        image = Image.open('./food41/images/'+ line + '.jpg', 'r')
        new_image = standarize(image, 300)
        resized_array.add(new_image)
    #img = toMatrix(train_foods[0])
    #classifier = Sequential()

main()

# CNN stuff
#from keras.models import Sequential
#from keras.layers import Convolution2D
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from skimage import filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value

#This is all of our images we train off of
#Open them and make it pretty
trainData = open('./food41/meta/meta/train.txt', 'r')
train_foods = trainData.read().splitlines()

#Takes the image and turns it into multidimensional list
#that is based off of the image's width and height
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

def main():
    print(train_foods)
    print("HELLOLOLOL0 CHIKCHENS")
    img = toMatrix(train_foods[0])
    #classifier = Sequential()

main()

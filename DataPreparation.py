# import libraries
import numpy as np
import pickle as pk
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
# image processing
from keras.preprocessing.image import ImageDataGenerator


# loading data
image_paths = 'images'

train_images = pd.read_csv('images/train.csv')
test_images = pd.read_csv('images/test.csv')
samples = pd.read_csv('sample_submission.csv')

# total images
print('image in train',len(train_images))
print('image in test',len(test_images))
print('total images',len(train_images) + len(test_images))

# function visualizer
def visualizer(imgs, figs = (15, 15), c=None):
    _,a = plt.subplot(5, 5, figsize = figs)

    for i, j in enumerate(a.flat):
        j.imshow(imgs[i], cmap = c)

# plot the first and second images
for image,label in zip(train_images.iloc[:,0],train_images.iloc[:,1]):

    img_path = os.path.join(image_paths,image)
    img  = Image.open(img_path)

    plt.imshow(img)
    plt.figure()
    img = img.filter(ImageFilter.MedianFilter)

    plt.imshow(img)
    print(img.size,img.mode,img.format)
    print(type(img))

    break

# gray the image
for image, label in zip(train_images.iloc[:, 0], train_images.iloc[:, 1]):
    img_path = os.path.join(image_paths, image)
    img = Image.open(img_path)

    gray = ImageOps.grayscale(img)

    plt.imshow(gray, cmap='gray')

    print(img.size, img.mode, img.format)
    print(type(gray))

    break

# create train and label lists
train  = []
labels = []
for image,label in zip(train_images.iloc[:,0],train_images.iloc[:,1]):

    img_path = os.path.join(image_paths,image)
    img  =  Image.open(img_path)
    img  = img.filter(ImageFilter.MedianFilter)
    train.append(np.asarray(img))
    labels.append(label)
#    plt.imshow(img)

# subplot
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))

for index, i in enumerate(ax.flatten()):
    i.imshow(train[index])
    i.set_title(labels[index])

    i.set_xticks([])
    i.set_yticks([])















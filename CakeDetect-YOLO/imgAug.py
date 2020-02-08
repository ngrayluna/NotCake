#!/bin/usr/python3
# -------------------------------------------------------------------
# Author: Noah G. Luna
#
# Description:
# ------------
# WIP for data augmentation and image generator for training.
# Completed: NO
# -------------------------------------------------------------------

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from processing import parse_annotation

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# TODO 
# Finish Class ImageGenerator
# class ImageDataGenerator:
#     def __init__(self, image, batch_size, numClass = 1, shuffle = True):
        
#         self.images     = image
#         self.batch_size = batch_size
#         self.numClass   = numClass
#         self.shuffle    = shuffle
        


def main():

	# Define parameters of Training Data
	train_image_folder = './images/train/'
	train_annot_folder = './images/train_annot/'

	# Define parameters of the validation data
	valid_image_folder 	= './images/valid/'
	valid_annot_folder 	= './images/valid_annot/'

	labels 	 = list(['cake'])

	###############################
	#   Parse the annotations / read in train and valid data
	###############################

	# parse annotations of the training set
	trainInfo, _ = parse_annotation(ann_dir = train_annot_folder, 
	    img_dir = train_image_folder, 
	    labels = labels)

	# parse annotations of the validation set
	validInf, _ = parse_annotation(ann_dir = valid_annot_folder, 
	    img_dir = valid_image_folder, 
	    labels = labels)

	# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
	# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
	# image.
	sometimes = lambda aug: iaa.Sometimes(0.5, aug)

	augPipe = iaa.Sequential(
	    [
	        # apply the following augmenters to most images
	        #iaa.Fliplr(0.5), # horizontally flip 50% of all images
	        #iaa.Flipud(0.2), # vertically flip 20% of all images
	        #sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
	        sometimes(iaa.Affine(
	            #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
	            #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
	            rotate=(-10, 10), # rotate by -45 to +45 degrees
	            #shear=(-5, 5), # shear by -16 to +16 degrees
	            #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
	            #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
	            #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
	        )),
	        # execute 0 to 5 of the following (less important) augmenters per image
	        # don't execute all of them, as that would often be way too strong
	        iaa.SomeOf((0, 5),
	            [
	                iaa.OneOf([
	                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
	                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
	                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
	                ]),
	                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
	                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
	                iaa.OneOf([
	                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
	                ]),
	                #iaa.Invert(0.05, per_channel=True), # invert color channels
	                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
	                iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
	                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
	                #iaa.Grayscale(alpha=(0.0, 1.0)),
	                #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
	                #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
	            ],
	            random_order=True
	        )
	    ],
	    random_order=True
	)

	############################### TMP ###############################
	# Check an example to make sure it works properly
	imgCheck = 3
	imPract = cv2.imread(os.path.join(trainInfo[imgCheck]['filename']))

	# Convrt to images from BGR -> RGB since we are going to plot this
	# using Matplotlib
	imPract = cv2.cvtColor(imPract, cv2.COLOR_BGR2RGB)

	rows, cols = imPract.shape[:2]

	# Let's grab one of the annotations and plot them.
	currAnnot = trainInfo[imgCheck]
	xminBox = currAnnot['object'][0]['xmin']
	yminBox = currAnnot['object'][0]['ymin']
	xmaxBox = currAnnot['object'][0]['xmax']
	ymaxBox = currAnnot['object'][0]['ymax']

	x1 = xminBox
	x2 = xmaxBox
	y1 = yminBox
	y2 = ymaxBox

	bbs = BoundingBoxesOnImage([
	    BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2)],shape=imPract.shape)

	ia.imshow(bbs.draw_on_image(imPract, size=2))


	# Augment Data and modify bounding box
	image_aug, bbs_aug = augPipe(image=imPract, bounding_boxes=bbs)
	ia.imshow(bbs_aug.draw_on_image(image_aug, size=2))


if __name__ == '__main__':
	main()

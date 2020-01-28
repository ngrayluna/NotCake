#!/bin/usr/python3
# -------------------------------------------------------------------
# Author: experiencor/keras-yolo2 with  modification by Noah G. Luna
#
# Description:
# ------------
# 
# -------------------------------------------------------------------


import os
import argparse
import numpy as np
from preprocessing import parse_annotation
from frontend import YOLO 
import json



## TO DO ##

# See if I am using weights from ImageNet or the onedefined on the GitHub one (line 44)
# Find out about anchors (see line 37)
# Understand the BaseFeatureExtractor Class
# Understand loss function in YOLO paper./Read Paper



def main(args):

	###############################
	#   Define Parameters of Model
	architecture 		= "MobileNetFeatureV2"  # "Tiny Yolo" or "MobileNet" or "ResNet"
	input_size 			= 400
	anchors 			= #
	max_box_per_image 	= 5
	labels 				= ['cake']	

	# Define parameters of Training Data
	train_image_folder = './images/train/'
	train_annot_folder = './images/train_annot/'

	train_times		 	= 8 # the number of time to cycle through the training set, useful for small datasets
	pretrained_weights 	= '' # path of the pretrained weights, OK to start from scratch
	batch_size 			= 16 # number of images to read in each batch
	learning_rate		= 1e-4 # the base learning rate of the default Adam rate scheduler
	nb_epochs 			= 50 # number of epochs
	warmup_epochs 		= 2 # number of initial epochs during which the size of the 5 boxes in each cell is forced to match the sizes of the 5 anchors, this trickseems to improve emperically

	object_scale 		= 5.0 # determine how much to penalize wrong prediction of confidence of object predictors
	no_object_scale 	= 1.0 # determine how much to penalize wrong prediction of confidence of non-object predictors
	coord_scale			= 1.0 # determine how much to penalize wrong position and size predictions (x, y, w, h)
	class_scale			= 1.0 # determine how much to penalize wrong class prediction

	saved_weights_name 	= 'best_weights.h5' # name of the output file
	debug				= True

	# Define parameters of the validation data
	valid_image_folder 	= './images/valid/'
	valid_annot_folder 	= './images/valid_annot/'
	valid_times 		= 

	###############################
	#   Parse the annotations 
	###############################

	# parse annotations of the training set
	train_imgs, train_labels = parse_annotation(ann_dir = train_annot_folder, 
		img_dir = train_image_folder, 
		labels = labels)

	# parse annotations of the validation set
	valid_imgs, valid_labels = parse_annotation(ann_dir = valid_annot_folder, 
		img_dir = valid_image_folder, 
		labels = labels)    

	###############################
	#   Construct the model
	###############################

	# Define yolo instance from YOLO Class
	yolo = YOLO(backend 			= architecture,
				input_size 			= input_size,
				labels 				= labels,
				max_box_per_image 	= max_box_per_image,
				anchors 			= anchors)


	###############################
	#   Load the pretrained weights (if any)
	###############################

	if os.path.exists(pretrained_weights):
		print("Loading pre-trained weights in", pretrained_weights)
		yolo.load_weights(pretrained_weights)


	###############################
	#   Start Training
	###############################

	yolo.train(train_imgs  = train_imgs,
		valid_imgs         = valid_imgs,
		train_times        = train_times,
		valid_times        = valid_times,
		nb_epochs          = nb_epochs,
		learning_rate      = learning_rate,
		batch_size         = batch_size,
		warmup_epochs      = warmup_epochs,
		object_scale       = object_scale,
		no_object_scale    = no_object_scale,
		coord_scale        = coord_scale,
		class_scale        = class_scale,
		saved_weights_name = saved_weights_name,
		debug              = debug)




if __name__ == '__main__':
	main()
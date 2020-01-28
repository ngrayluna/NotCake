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

# Create config file
# See if I am using weights from ImageNet or the onedefined on the GitHub one
# Get tiny yolo weights
# Understand the BaseFeatureExtractor Class
# Understand loss function in YOLO paper.
# Read YOLO paper

# Make command line-friendly
argparser = argparse.ArgumentParser(description="Train and validate YOLO_V2")

argparser.add_argument('-c','--conf',help='path to configuration file')


def main(args):
	config_path = args.conf 

	with open(config_path) as config_buffer:
		conf = json.loads(config_buffer.read())


	###############################
	#   Parse the annotations 
	###############################


	###############################
	#   Construct the model
	###############################

	# Define yolo instance from YOLO Class
	yolo = YOLO(backend 			= config['model']['backend'],
				input_size 			= config['model']['input_size'],
				labels 				= config['model']['labels'],
				max_box_per_image 	= config['model']['max_box_per_image'],
				anchors 			= config['model']['anchors'])


	###############################
	#   Load the pretrained weights (if any)
	###############################



	###############################
	#   Start Training
	###############################

    yolo.train(train_imgs  = train_imgs,
		valid_imgs         = valid_imgs,
		train_times        = config['train']['train_times'],
		valid_times        = config['valid']['valid_times'],
		nb_epochs          = config['train']['nb_epochs'], 
		learning_rate      = config['train']['learning_rate'], 
		batch_size         = config['train']['batch_size'],
		warmup_epochs      = config['train']['warmup_epochs'],
		object_scale       = config['train']['object_scale'],
		no_object_scale    = config['train']['no_object_scale'],
		coord_scale        = config['train']['coord_scale'],
		class_scale        = config['train']['class_scale'],
		saved_weights_name = config['train']['saved_weights_name'],
		debug              = config['train']['debug'])




if __name__ == '__main__':
	args = argparser.parse_args()
	main(args)
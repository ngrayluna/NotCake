#!/bin/usr/python3
# -------------------------------------------------------------------
# Author: experiencor/keras-yolo2 with  modification by Noah G. Luna
#
# Description:
# ------------
# YOLOv2 with MobileNet backend work-in-progress.
# -------------------------------------------------------------------
import os
import argparse
import numpy as np
from processing import parse_annotation
from frontend import YOLO 
import json

####### TO DO #########

# Understand loss function in YOLO paper./Read Paper

# Finish commenting Class in frontend.py and backend.py

# https://github.com/experiencor
# https://github.com/makatx/YOLO_ResNet/blob/master/model_continue_train.py


####### THINGS TO UPDATE ON ORIGINAL #######
#tf.Print -> tf.print

#seen = tf.assign_add(seen, 1.) # depecrated TensorFlow v1
#seen.assign_add(1.0)
##############


def main(args):
	config = args.config

	
	# parse annotations of the training set
	train_imgs, train_labels = parse_annotation(ann_dir = config['train']['train_annot_folder'], 
		img_dir = config['train']['train_image_folder'], 
		labels = config['model']['labels'])

	# parse annotations of the validation set
	valid_imgs, valid_labels = parse_annotation(ann_dir = config['valid']['valid_annot_folder'], 
		img_dir = config['valid']['valid_image_folder'], 
		labels = config['model']['labels'])    

	###############################
	#   Construct the model
	###############################

	# Define yolo instance from YOLO Class
    yolo = YOLO(backend         = config['model']['backend'],
			input_size          = config['model']['input_size'], 
			labels              = config['model']['labels'], 
			max_box_per_image   = config['model']['max_box_per_image'],
			anchors             = config['model']['anchors'])


	###############################
	#   Load the pretrained weights (if any)
	###############################

    if os.path.exists(config['train']['pretrained_weights']):
		print("Loading pre-trained weights in", config['train']['pretrained_weights'])
		yolo.load_weights(config['train']['pretrained_weights'])


	###############################
	#   Start Training
	###############################

	yolo.train(train_imgs         = train_imgs,
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
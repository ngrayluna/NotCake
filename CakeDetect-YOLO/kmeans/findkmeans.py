#!/bin/usr/python3
# -------------------------------------------------------------------
# Author: Noah G. Luna
#
# Description:
# ------------
# k-means to find anchor boxes.
# -------------------------------------------------------------------

import os
import numpy as np
from kmeans import getBoxes, kmeansAlg, avg_iou, parse_bboxes

## TODO: Create argparse to specify directory location of training data
## TODO: Save output into a .txt file?
## Get rid of labels variable... make config file instead?

def main():
	# Number of clusters:
	clusters = 5

	labels = list(['cake'])

	# parse data to get xmin, ymin, etc.
	# Define parameters of Training Data
	train_image_folder = './images/train/'
	train_annot_folder = './images/train_annot/'

	# Define parameters of the validation data
	valid_image_folder 	= './images/valid/'
	valid_annot_folder 	= './images/valid_annot/'

	###############################
	#   Parse the annotations 
	###############################

	# parse annotations of the training set
	trainInfo, _ = parse_bboxes(ann_dir = train_annot_folder, 
		img_dir = train_image_folder, 
		labels = labels)

	# parse annotations of the validation set
	validInfo, _ = parse_bboxes(ann_dir = valid_annot_folder, 
		img_dir = valid_image_folder, 
		labels = labels)


	# Load data set
	data   = getBoxes(trainInfo + validInfo)

	###############################
	#   k-means Algorithm
	###############################

	# Run k-means
	output = kmeansAlg(data, k=clusters)

	# print results
	print("Accuracy: {:.2f}%".format(avg_iou(data, output) * 100))
	print("Boxes:\n {}".format(output))

	ratios = np.around(output[:, 0] / output[:, 1], decimals=2).tolist()
	print("Ratios:\n {}".format(sorted(ratios)))	

if __name__ == '__main__':
	main()


	###############################
	#   Example OutPut:
	###############################
	# Accuracy: 84.38%
	# Boxes:
	#  [[311 221]
	#  [378 371]
	#  [248 271]
	#  [215 177]
	#  [332 303]]
	# Ratios:
	#  [0.92, 1.02, 1.1, 1.21, 1.41]
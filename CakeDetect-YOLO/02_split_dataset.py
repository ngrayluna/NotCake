#!/bin/usr/python3
# -------------------------------------------------------------------
# Author: Noah G. Luna 
#
# Description:
# ------------
# Splits images and their associated annotations into a training and 
# validation set.
#
# Returns:
# --------
# train_image, train_annot, valid_image, and a valid_annot folders
# with their associated images and bounding box annotations in VOC
# format.
# -------------------------------------------------------------------

import os
import argparse
import random
import numpy as np
from shutil import copyfile


def checkIfDirExists(directory:str):
	"""
	Check if directory exists, if not, create it.

	Parameters:
	---------
	"""
	if not os.path.exists(directory):
		os.mkdir(directory)


def createTrainValDir(imageDirectory:str):
	"""
	Create directories to store the train, validation, and annotated
	image files.
	"""
	directoryNames = ['train', 'valid', 'train_annot', 'valid_annot']

	directoryPaths = [os.path.join(imageDirectory, dir) for dir in directoryNames]

	for directory in directoryPaths:
		checkIfDirExists(directory)

	return directoryPaths



def copyMoveFiles(fileNames:str, inputDirectory:str, outputDirectory:str):
	"""
	Copy files from one directory into another
	"""

	for file in fileNames:
		source = os.path.join(inputDirectory, file)
		dst = os.path.join(outputDirectory, file)
		copyfile(source, dst)


def main()
	############################################
	# TO DO
	# store name of image file in config file?
	###################################################

	# Image directory
	imageDir = './images'

	# Name of directory where processed images are stored:
	procImgs = './images/processed'

	# Get file names and make sure it is a file and not a directory
	fileNames = [name for name in os.listdir(procImgs) if os.path.isfile(os.path.join(procImgs, name))]

	# Shuffle file names
	random.shuffle(fileNames)

	# Find out how many files are in image directory
	numFiles = len(fileNames)

	# Declare split ratio
	##########################################
	## TO DO: Put into configuration file
	split_train = .75
	##########################################

	# Compute split ratio
	trainSize = int(np.floor(numFiles * split_train))

	# Split file name
	trainFileNames = fileNames[:trainSize]
	validFileNames = fileNames[trainSize:]

	# Create directory(s) to store train, val, and annotation files
	directoryPaths = createTrainValDir(imageDir)

	# Read in 'x' number of files int train, validation folders
	copyMoveFiles(trainFileNames, procImgs, directoryPaths[0])
	copyMoveFiles(validFileNames, procImgs, directoryPaths[1])

	print("Copying complete")

	#########################################################################
	# TO DO Read in 'x' number of files into train_annot, valid_annot folders


if __name__ == '__main__':
	main()


#!/bin/usr/python3
# -------------------------------------------------------------------
# Author: Noah G. Luna 
#
# Description:
# ------------
# Splits images and their associated annotations into a training and 
# validation set. Creates and stores images and their associated 
# annotations into separate folders (four in total).
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

	Parameters:
	---------
	"""
	directoryNames = ['train', 'valid', 'train_annot', 'valid_annot']

	directoryPaths = [os.path.join(imageDirectory, dir) for dir in directoryNames]

	for directory in directoryPaths:
		checkIfDirExists(directory)

	return directoryPaths



def copyMoveFiles(fileNames:str, inputDirectory:str, outputDirectory:str):
	"""
	Copy image files from one directory into another.

	Parameters:
	---------
	"""

	for file in fileNames:
		
		# input directory
		source = os.path.join(inputDirectory, file)

		# Where to store the files
		dst = os.path.join(outputDirectory, file) 
		
		# Copy image to distination folder.
		copyfile(source, dst)


def copyMoveAnnotations(fileNames:str, inputDirectory:str, outputDirectory:str):
	"""
	Copy annotation files from one directory to another.

	Parameters:
	---------
	"""

	for file in fileNames:

		# First, grab file number
		annotPrefix = file[:-4]
		
		# Paste with correct output file name
		annot_file = annotPrefix + '.xml'

		source = os.path.join(inputDirectory, annot_file)

		# Copy image to destination folder.
		if os.path.split(outputDirectory)[-1] == 'valid_annot':
			trueOutputDir = './images/valid_annot/'

			# Path to file with annotations for validation images
			dst = os.path.join(trueOutputDir, annot_file)

			# Copy annotation to separate folder
			copyfile(source, dst)

		else:
			trueOutputDir = './images/train_annot/'

			# Path to file with annotations for training images
			dst = os.path.join(trueOutputDir, annot_file)

			# Copy annotation to separate folder
			copyfile(source, dst)


# Create the parser
argparser = argparse.ArgumentParser(
	description='Split data set into separate directories.')

# Add arguments
argparser.add_argument(
	'path',
	 type = str,
	 help ='path to boundary box annotations')


def main(args):
	############################################
	# TO DO
	# store name of image file in config file?
	###################################################

	# Image directory
	imageDir = './images'

	# Name of directory where processed images are stored:
	procImgs = './images/processed'

	# Name of directory where the annotations are:
	annotDirectory = args.path
	#annotDirectory = './images/annotations/'

	# Get file names and make sure it is a file and not a directory
	fileNames = [name for name in os.listdir(procImgs) if os.path.isfile(os.path.join(procImgs, name))]

	# Shuffle file names
	random.shuffle(fileNames)

	# Find out how many files are in image directory
	numFiles = len(fileNames)

	# Declare split ratio
	##########################################
	## TO DO: Put into configuration file
	split_train = .80
	##########################################

	# Compute split ratio
	trainSize = int(np.floor(numFiles * split_train))

	# Split file name
	trainFileNames = fileNames[:trainSize]
	validFileNames = fileNames[trainSize:]

	# Create directory(s) to store train, val, and annotation files
	directoryPaths = createTrainValDir(imageDir)

	# Read in 'x' number of files into train, validation folders
	copyMoveFiles(trainFileNames, procImgs, directoryPaths[0])
	copyMoveFiles(validFileNames, procImgs, directoryPaths[1])

	# Read in 'x' number of files into train and validation annotation folders
	copyMoveAnnotations(trainFileNames, annotDirectory, directoryPaths[2])
	copyMoveAnnotations(validFileNames, annotDirectory, directoryPaths[3])

	print("Copying complete")


if __name__ == '__main__':

	# Exectute parse_args()
	args = argparser.parse_args()
	main(args)

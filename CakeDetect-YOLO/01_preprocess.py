#!/bin/usr/python3
# -------------------------------------------------------------------
# Author: Noah G. Luna 
#
# Description:
# ------------
# Reads in JPEG files and resizes them with specified width and
# height. Modified images are saved in a separate directory. Default
# directory is: ./imgs_processed.
#
# -------------------------------------------------------------------

import os
import sys
import argparse
from PIL import Image


def checkIfDirExists(directory:str):
	"""
	Check if directory exists, if not, create it.

	Parameters:
	---------
	"""
	if not os.path.exists(directory):
		os.mkdir(directory)


def readFile(directory:str, file:str):
	"""
	Create file path by concatenating file directory
	and file name into one string.

	Parameters:
	---------

	Return:
	-------

	"""
	filePath = os.path.join(directory, file)

	return filePath


def resizeImage(imageFile:str, width:int, height:int, outfile:str):
	"""
	Resize image with specified width and height.

	Parameters:
	---------

	"""
	try:
		im = Image.open(imageFile)
		print("Resizing...")
		im = im.resize((height, width))
		im.save(outfile, "JPEG")
		print("Saving modified image.")
	except Exception as e:
		print("Trouble modifying image. Skipping.\n")


# Create the parser
argparser = argparse.ArgumentParser(
	description='Process images dataset for training.')

# Add arguments
argparser.add_argument(
	'path',
	 type = str,
	 help ='path to raw input images')



def main(args):

	# Get directory of raw images to be processed.
	inDir = args.path
	if not os.path.isdir(inDir):
		print('The path specified does not exist')
		sys.exit()

	# Store processed images in separate directory.
	outDir = './images/processed/' 


	####### TO DO: Add to config file ########
	height = 400 
	width = 400
	##########################################


	# Check if output directory exists [do this once]
	checkIfDirExists(outDir)

	# Read in all raw images and save to output directory
	for inFile in os.listdir(inDir):
		
		# Read file in.
		file = readFile(inDir, inFile)
		print("Reading in {}".format(os.path.split(file)[-1]))

		# Create output file name
		file_num = os.path.split(file)[-1].split('.')[0]
		outfile = os.path.join(outDir, 'cake_'+ file_num + '.jpg')

		#### TO DO Check that output file does not exist in directory ###
		
		
		# Change image size #
		resizeImage(file, width, height, outfile)
		
	print("\nImage processing complete.")


if __name__ == '__main__':
	# Exectute parse_args()
	args = argparser.parse_args()
	main(args)

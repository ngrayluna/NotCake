#!/bin/usr/python3
# -------------------------------------------------------------------
# Author: Noah G. Luna 
#
# Description:
# ------------
# Script to check:
# + Image opens without fail (i.e. not corrupted)
# + Image has proper height and width and hieght. 
# + Each .jpg image has an associated annotated bounded box .xml file
#
# Returns
# -------
# A missingAnnotations.txt file containing the name of .jpg which
# do not have bounding box annotations .xml file.
# -------------------------------------------------------------------

import os
from PIL import Image


def checkImageCanOpen(imageFile:str):
	"""
	Check if we can open the image. If not, it's corrupted.
	Returns the image object (we successfully opened) and
	a bool stating if succesful opening file.

	Parameters:
	---------

	"""
	try:
		# Try to open the image.
		im = Image.open(imageFile)		

		# Return it works
		openSuccesful = True

		return im, openSuccesful
	
	except Exception as e:
		# Remove image 
		os.remove(imageFile)

		# Return emtpty string
		im = ''

		# Return it failed
		openSuccesful = False
		
		return im, openSuccesful


def main():
	####### TO DO: Add to config file ########
	height = 400 
	##########################################

	# Path to where processed images are stored:
	procImgs = './images/processed'	

	# Path to the bounding box annotations
	xmlDir = './images/annotations'

	# Try to open file, if error delete
	for inFile in os.listdir(procImgs):
		
		# Path to processed image files
		filePath = os.path.join(procImgs, inFile)

		# Check if we can open the image:
		im, openStatus = checkImageCanOpen(filePath)

		if openStatus:
			# Check file size, if zero remove
			imWidth, imHeight = im.size
			if (imWidth != height) or (imHeight != height):
				im.close()
				os.remove(filePath)

		# Let's check the associated .xml file is also there
		xmlFile = os.path.split(inFile)[-1].split('.')[0] + '.xml'

		# Path to .xml file with bounding box annotations
		xmlPath = os.path.join(xmlDir, xmlFile)

		# If not, store the name of the image file in a .txt file
		if not os.path.isfile(xmlPath):

			# Open file and write name of .jpg with missing annotated file
			missingTxt = open('missingAnnotations.txt', 'a')
			missingTxt.write(inFile + '\n')
			missingTxt.close()


	print("Check Complete.")

	# Check if any files where missing.
	isEmpty = os.stat('missingAnnotations.txt').st_size == 0
	if not isEmpty:
		print('Missing .xml annotation for .jpg file(s).\n')
		print('Please refer to "missingAnnotations.txt" file to view .jpg files which are missing annotations.')


if __name__ == '__main__':
	main()
#!/bin/usr/python3
# -------------------------------------------------------------------
# Author: Noah G. Luna 
#
# Description:
# ------------
# Script to check you can open the image without fail, 
# checks it has proper height and width and hieght. 
#
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

		# Give emtpty string back
		im = ''

		# Return it failed
		openSuccesful = False
		
		return im, openSuccesful


def main():
	####### TO DO: Add to config file ########
	height = 400 
	##########################################

	# Name of directory where processed images are stored:
	procImgs = './images/processed'	

	# Try to open file, if error delete
	for inFile in os.listdir(procImgs):
		
		filePath = os.path.join(procImgs, inFile)

		# Check if we can open the image:
		im, openStatus = checkImageCanOpen(filePath)

		if openStatus:
			# Check file size, if zero remove
			imWidth, imHeight = im.size
			if (imWidth != height) or (imHeight != height):
				im.close()
				os.remove(filePath)

	print("Check Complete.")

if __name__ == '__main__':
	main()
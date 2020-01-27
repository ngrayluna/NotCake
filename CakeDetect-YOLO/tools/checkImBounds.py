#!/bin/usr/python3
# -------------------------------------------------------------------
# Author: Noah G. Luna 
#
# Description:
# ------------
# Script to check:
# Reads in a file an its associated annoted .xml file to make sure
# it saved correctly. - A sanity check.
#
# Returns
# -------
# Directly plots image.
# 
# Optional: Save image.
# -------------------------------------------------------------------


import os
import cv2
import argparse
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def parse_annotations(annotPath:str, imagePath:str):

	imgsFound = []
	seen_labels = {}
	labels = []

	for annot in sorted(os.listdir(annotPath)):

		# Dictionary to store information from .xml file
		imgInfo = {'object':[]}

		# Path to annotation file
		annotFile = os.path.join(annotPath, annot)

		# Initialize tree with ET
		tree = ET.parse(annotFile)

		# Go through each tag
		for elem in tree.iter():
			if 'filename' in elem.tag:
			    imgInfo['filename'] = os.path.join(imagePath, elem.text)
			if 'width' in elem.tag:
			    imgInfo['width'] = int(elem.text)
			if 'height' in elem.tag:
			    imgInfo['height'] = int(elem.text)

			if ('object' in elem.tag) or ('part' in elem.tag):
				obj = {}

				# Let's look at the attributes in the Object tag
				for attrib in list(elem):

					# Name information
					if 'name' in attrib.tag:
						obj['name'] = attrib.text

						if obj['name'] in seen_labels:
						    seen_labels[obj['name']] += 1
						else:
							seen_labels[obj['name']] = 1

						if len(labels) > 0 and obj['name'] not in labels:
							break
						else:
							imgInfo['object'] += [obj]


					# Bounding box information
					if 'bndbox' in attrib.tag:
						for dim in list(attrib):
							if 'xmin' in dim.tag:
							    obj['xmin'] = int(round(float(dim.text)))
							if 'ymin' in dim.tag:
							    obj['ymin'] = int(round(float(dim.text)))
							if 'xmax' in dim.tag:
							    obj['xmax'] = int(round(float(dim.text)))
							if 'ymax' in dim.tag:
							    obj['ymax'] = int(round(float(dim.text)))

		# Make sure we don't have an empty file
		if len(imgInfo['object']) > 0:
			imgsFound += [imgInfo]

	return imgsFound, seen_labels


# Create the parser
argparser = argparse.ArgumentParser(
	description='Checks image annotation saved properly')

# Add arguments
argparser.add_argument(
	'path',
	 type = str,
	 help ='path to image')



def main(args):

	# Where file image is
	imgFile = args.path

	# Path image and annotation
	imagePath = '../images/train/'
	annotPath = '../images/train_annot/'
	#imgFile = '../images/train/cake_00000004.jpg'

	# Get informaton from .xml files
	imgsFound, seen_labels = parse_annotations(annotPath, imagePath)


	# Let's use cv2
	imPract = cv2.imread(os.path.join(imgFile))
	height, width = imPract.shape[0], imPract.shape[1]

	# Convrt to images from BGR -> RGB since we are going to plot this
	# using Matplotlib
	imPract = cv2.cvtColor(imPract, cv2.COLOR_BGR2RGB)


	# Let's grab one of the annotations and plot them.
	currAnnot = imgsFound[3]
	xminBox = currAnnot['object'][0]['xmin']
	yminBox = currAnnot['object'][0]['ymin']
	xmaxBox = currAnnot['object'][0]['xmax']
	ymaxBox = currAnnot['object'][0]['ymax']

	# represents the top left corner of rectangle
	start_point = (xminBox,yminBox)

	# represents the bottom right corner of rectangle 
	end_point = (xmaxBox,ymaxBox)

	# Color in BGR
	color = (0,255,0)

	# thickness of the box
	thickness = 3

	# Using cv2.rectangle() method 
	image = cv2.rectangle(imPract, start_point, end_point, color, thickness)

	# Plot image
	plt.imshow(image)
	plt.show()

	# Save image
	#cv2.imwrite('./imgTest.png', image)

if __name__ == '__main__':
	args = argparser.parse_args()
	main(args)


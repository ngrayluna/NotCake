#!/bin/usr/python3
# -------------------------------------------------------------------
# Author: Noah G. Luna 
#
# Description:
# ------------
# Various tools for reading in data for training.
# 
# -------------------------------------------------------------------


import os
import xml.tree.ElementTree as ET 

def parse_annotations(annotDir:str, imgDir:str):

	for annot in os.listdir(annotDir):

		tree = ET.parse(annotDir)
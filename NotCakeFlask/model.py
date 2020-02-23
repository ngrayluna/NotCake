#!/bin/usr/python3
from keras.applications.inception_v3 import *
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
import json
global graph



def predict(path):
	model = InceptionV3(weights='imagenet')
	
	img = image.load_img(path, target_size=(299, 299))
	xy = image.img_to_array(img)
	xy = np.expand_dims(xy, axis=0)
	xy = preprocess_input(xy)

	preds = model.predict(xy)
	preds = decode_predictions(preds, top=3)[0]
	acc = []
	classes = []
	
	for x in preds:
		acc.append(x[2])
		classes.append(x[1])

	del model, img, xy, preds

	return classes, acc
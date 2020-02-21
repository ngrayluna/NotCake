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
	graph = tf.get_default_graph()

	img = image.load_img(path, target_size=(299, 299))
	xy = image.img_to_array(img)
	xy = np.expand_dims(xy, axis=0)
	xy = preprocess_input(xy)


	with graph.as_default():
		preds = model.predict(xy)
		preds = decode_predictions(preds, top=3)[0]
		acc = []
		classes = []
	
	for x in preds:
		acc.append(x[2])
		classes.append(x[1])

	return classes, acc
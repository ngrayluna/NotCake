#!/bin/usr/python3
# -------------------------------------------------------------------
# Author: Noah G. Luna 
#
# Description:
# ------------
#
# -------------------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Reshape, Lambda


## TO DO ##

# See if I am using weights from ImageNet or the onedefined on the GitHub one
# Get tiny yolo weights
# Understand the BaseFeatureExtractor Class




TINY_YOLO_BACKEND_PATH = "tiny_yolo_backend.h5"
#MOBILENET_BACKEND_PATH = "mobilenet_backend.h5"




class BaseFeatureExtractor(object):
	"""
	Checks,,,,
	"""

	# Will be defined in each subclass
	def __init__(self, input_size):
		raise NotImplementedError("error message")

	# Will be defined in each subclass
	def normalize(self, image):
		raise NotImplementedError("error message")

	def get_output_shape(self):
		return self.feature_extractor.get_output_shape_at(-1)[1:3]

	def extract(self, input_image):
		return self.feature_extractor(input_image)



class TinyYoloFeature(BaseFeatureExtractor):
	"""
	Defines Tiny YOLO architecture for feature extracting.
	"""

	def __init__(self, input_size):
		input_image = Input(shape=(input_size, input_size), 3)

		# Layer 1
		x = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1),\
			padding='same', name='conv_1', use_bias=False)(input_image)
		x = BatchNormalization(name='norm_1')(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2,2))(x)

		# Layer 2 - 5
        for i in range(0,4):
			x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same',\
				name='conv_' + str(i+2), use_bias=False)(x)
			x = BatchNormalization(name='norm_' + str(i+2))(x)
			x = LeakyReLU(alpha=0.1)(x)
			x = MaxPooling2D(pool_size=(2, 2))(x)

		# Layer 6
		x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1),\
			padding='same', name='conv_6', use_bias=False)(x)
		x = BatchNormalization(name='norm_6')(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

		# Layers 7 - 8
        for i in range(0,2):
			x = Conv2D(1024, (3,3), strides=(1,1), padding='same',\
				name='conv_' + str(i+7), use_bias=False)(x)
			x = BatchNormalization(name='norm_' + str(i+7))(x)
			x = LeakyReLU(alpha=0.1)(x)

		self.feature_extractor = Model(input_image, x)
		self.feature_extractor.load_weights(TINY_YOLO_BACKEND_PATH)

	def normalize(self, image):
		return image / 255.



class MobileNetFeature(BaseFeatureExtractor):
	"""
	Defines MobileNet architecture for feature extracting. In this
	model we are using the MobileNetV2 with weights trained on the 
	ImageNet data set.
	"""
	def __init__(self, input_size):
		input_image = Input(shape=(input_size, input_size, 3))

		mobilenet = MobileNetV2(input_shape=(224, 224, 3),\
			include_top = False, weights = 'imagenet')
		# mobilenet.load_weights(MOBILENET_BACKEND_PATH)

		x = mobilenet(input_image)

		# Define instance feature extractor model and set
		# as attribue of class
		self.feature_extractor = Model(input_image, x)


    def normalize(self, image):
		image = image / 255.
		image = image - 0.5
		image = image * 2.

		return image



class YOLO(object):
	"""
	DocString for Class
	"""
	def __init__(self, backend,
		input_size,
		labels,
		max_box_per_image,
		anchors)

	self.input_size = input_size
	self.labels = labels
	self.nb_class = len(self.labels)	
	self.nb_box = len(anchors)
	self.anchors = anchors
	self.class_wt = np.ones(self.nb_class, dtype = 'float32')
	self.max_box_per_image = max_box_per_image


	############################################
	# Make the model
	############################################


	# Make the feature extractor layers
	input_image = Input(shape=(self.input_size, self.input_size, 3))
	self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4))


	# Set the backend. This is our feature extractor/base model
	if backend == 'MobileNet':
		self.feature_extractor = MobileNetFeature(self.input_size)
	elif backend == 'Tiny Yolo':
		self.feature_extractor = TinyYoloFeature(self.input_size)
	else:
		raise Exception('Architecture is not supported. Please see list of supported architectures.')


	# Print the feature extractor shape as defined in the Class specified above
	print(self.feature_extractor.get_output_shape())

	# Assign feature extractor shape to instance variabkes so we can use this as input
	# to our YOLO object detector
	self.grid_h, self.grid_w = self.feature_extractor.get_output_shape() 
	features = self.feature_extractor.extract(input_image)


	####### YOLO #########
	# Last Layer
	# Define the object detection layer (the last portion of the model)
	output = Conv2D(self.nb_box * (4 + 1 + self.nb_class), (1,1), strides=(1,1), 
		padding='same', name='DetectionLayer', kernel_initializer='lecun_normal')(features)
	
	output = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))(output)

	# small hack to allow true_boxes to be registered when Keras build the model 
	output = Lambda(lambda args: args[0])([output, self.true_boxes])

	# Define the entire model, then get weights of the last layer
	self.model = Model([input_image, self.true_boxes], output)

	# Get last convolutional layer and its old weights
	layer = self.model.layers[-4]
	weights = layer.get_weights()

	# Randomize the weights of the last layer, use dimension of weights to define shape needed
	new_kernel = np.random.normal(size=weights[0].shape)/(self.grid_h*self.grid_w)
	new_bias   = np.random.normal(size=weights[1].shape)/(self.grid_h*self.grid_w)

	# Assign the new and randomized weights to the last layer
	layer.set_weights([new_kernel, new_bias])

	# print summary of the whole model
	self.model.summary()

	
	# Define a custom loss function

	# Define the train function

	# Define evaluate






def main():
	

	# Load data in (in a memory efficient way)

	# rescale images

	# Create the base model from the pre-trained model MobileNet V2
	yolo = YOLO(backend = '', input_size = '', labels = '', max_box_per_image = '', anchors = '')


	base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
		include_top=False,
		weights='imagenet')

	# Freeze convoltional base
	base_model.trainable = False

	# Let's look at the summary
	base_model.summary()

	# Add Classifier on top

	# Compile model

	# Train model

	# Plot learning curves

if __name__ == '__main__':
	main()
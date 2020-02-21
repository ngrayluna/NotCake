from flask import Flask, render_template, request
from model import predict
from keras.applications.inception_v3 import *
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
import json
global graph



app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
	return render_template("home.html")


@app.route("/about")
def about():
	tabName = 'About'
	return render_template("./about.html", title='NotCake - '+ tabName, page_name='About the Developer')


# Something is wrong with the 'methods=['POST']' thing.
# @app.route("/objectDetect", methods=['POST'])
# need to reset graph so it doesn't get weird....?

###### WILL PRINT RESULTS ######
# @app.route("/objectDetect")
# def objectDetect():

# 	tabName   = 'Cake Detection'
# 	page_name = 'Detection Cake'

# 	image_name = 'strawCake.jpg'
# 	path   = './imgTest/strawCake.jpg'

# 	classes, acc = predict(path)
# 	predPrint = "{} was found with {} confidence.".format(classes[0], acc[0])
# 	print(predPrint)

# 	return render_template("./objectDetect.html", title='NotCake - '+ tabName, page_name='Detection Cake', image_name=image_name, predPrint=predPrint)
# 	#return render_template("./objectDetect.html", title='NotCake - '+ tabName, page_name=page_name, image_name=image_name)



## TO DO ##
# one approch: take image, save, then read in
# second appraoch: take image, keep in memory, then upload <-

app.config["IMAGE_UPLOAD"] = './imgTest' # Store in a separate config file?

@app.route("/objectDetect", methods=["GET","POST"])
def objectDetect():

	tabName   = 'Cake Detection'
	page_name = 'Detection Cake'

	if request.method == "POST":
		if request.files:
			image = request.files["image"]

			print(request.files['image'])
			print(request.full_path)
			
			
			path = os.path.join(app.config["IMAGE_UPLOAD"], image.filename)
			print(path)

			# path = request.files["path"]
			classes, acc = predict(path)
			predPrint = "{} was found with {} confidence.".format(classes[0], acc[0])
			return render_template("./objectDetect.html", title='NotCake - '+ tabName, page_name='Detection Cake', predPrint=predPrint)
			
			#return render_template("./objectDetect.html", title='NotCake - '+ tabName, page_name=page_name, predPrint=image)


	if request.method == "GET":
		return render_template("./objectDetect.html", title='NotCake - '+ tabName, page_name=page_name)

	


if __name__ == '__main__':
	app.run(debug=True)

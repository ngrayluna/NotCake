from flask import Flask, render_template, request, url_for
from model import predict
from keras.applications.inception_v3 import *
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
import json
global graph



app = Flask(__name__)

app.config["IMAGE_UPLOAD"] = './imageDetect' # Store in a separate config file?


@app.route("/")
@app.route("/home")
def home():
	return render_template("home.html")


@app.route("/about")
def about():
	tabName = 'About'
	return render_template("./about.html", title='NotCake - '+ tabName, page_name='About')

@app.route("/howto")
def howto():
	tabName = 'How To'
	return render_template("./howto.html",  title='NotCake - '+ tabName, page_name='How To Use NotCake')


@app.route("/objectDetect", methods=["GET","POST"])
def objectDetect():

	tabName   = 'Cake Detection'
	page_name = 'Detection Cake'

	if request.method == "POST":

		# TO DO: check that we only get .jpg files
		if request.files:
			image = request.files["image"]
						
			path = os.path.join(app.config["IMAGE_UPLOAD"], image.filename)
			classes, acc = predict(path)
			
			isCake = False
			if acc[0] > 0.5 and classes[0]=='bakery':
				isCake = True
				predPrint = 'This is Cake!'
				cakeReturn = 'images/cafe-chocolate-cake.jpg' # TMP
			else:
				predPrint = 'The cake is a lie!'
				cakeReturn = '../static/images/cakelie.png'
			
			#print(request.files['image']) # print to terminal
			#print("Class predicted {} with accuracy {}".format(classes[0], acc[0]))

			imagePopUp = url_for('static', filename=cakeReturn)
 
			return render_template("./objectDetect.html", title='NotCake - '+ tabName, page_name='Detection Cake',\
				imagePopUp=imagePopUp, predPrint=predPrint)
	


	if request.method == "GET":
		return render_template("./objectDetect.html", title='NotCake - '+ tabName, page_name=page_name)

	


if __name__ == '__main__':
	app.run(debug=True)

from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
	return render_template("home.html")

@app.route("/about")
def about():
	tabName = 'About'
	return render_template("./about.html", title='NotCake - '+ tabName, page_name='About the Developer')

@app.route("/objectDetect", methods=['POST'])
def objectDetect():
	tabName = 'Cake Detection'

	

	return render_template("./objectDetect.html", title='NotCake - '+ tabName, page_name='Detection Cake')

if __name__ == '__main__':
	app.run(debug=True)

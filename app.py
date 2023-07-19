from flask import Flask, request, render_template
import json
import sqlite3
import subprocess

app = Flask(__name__)
app.debug = True

columns = ['No', 'EMAIL', 'NAME', 'STATUS', 'TIMESTAMP']

@app.route('/')
def index():
	return render_template('index.html', columns=columns)

@app.route('/open', methods=['POST'])
def open():
	print('classify')
	name = request.form['name']

	command = f"python classify_redline.py {name}"
	output = subprocess.check_output(command, shell=True, text=True)
	print(output)
	return output
	
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)
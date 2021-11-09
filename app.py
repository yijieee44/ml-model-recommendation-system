from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import pandas as pd

app = Flask(__name__)
type_list = ["Regression", "Classification"]
result = ""

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html', type_list=type_list, result=result, selected_type=type_list[0], selected_des="")


@app.route('/', methods=['POST'])
def submit():
    selected_type = request.form.get('selected_type')

    if 'file' not in request.files:
        result = "## Error: No file uploaded"
        return render_template('home.html', type_list=type_list, result=result, selected_type=selected_type, selected_des="")

    file = request.files['file']
    
    if file.filename == '':
        result = "## Error: No file selected"
        selected_des = ""

    if file:
        # generate result in string form here
        selected_des = file.filename
        selected_des = selected_des + " -- " + selected_type

        df = pd.read_csv(file.stream)
        result = df.to_string()
    
    return render_template('home.html', type_list=type_list, result=result, selected_type=selected_type, selected_des=selected_des)



if __name__ == "__main__":
    app.run
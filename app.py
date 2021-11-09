from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import warnings

from src.model.pycaret_compare import reg_compare, cla_compare, reg_create_graph, cla_create_graph

app = Flask(__name__)
type_list = ["Regression", "Classification"]
error_message = ""
warnings.filterwarnings("ignore")

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html', type_list=type_list, error_message=error_message, 
    selected_type=type_list[0], selected_des="", target_name="", graph_url="./static/images/graph.png")


@app.route('/', methods=['POST'])
def submit():
    selected_type = request.form.get('selected_type')
    target_name = request.form.get('target_name')
    error_message = ""

    if 'file' not in request.files:
        error_message = "## Error: No file uploaded"
        return render_template('home.html', type_list=type_list, error_message=error_message, selected_type=selected_type, selected_des="", target_name=target_name)

    file = request.files['file']
    
    if file.filename == '':
        error_message = "## Error: No file selected"
        selected_des = ""

    if file:
        # generate error_message in string form here
        selected_des = file.filename
        selected_des = selected_des + " -- " + selected_type
    
        df = pd.read_csv(file.stream)
        if selected_type == "Regression":
            best_model, result_df = reg_compare(df, target_name)
            reg_create_graph(result_df)
        elif selected_type == "Classification":
            best_model, result_df = cla_compare(df, target_name)
            cla_create_graph(result_df)
        
        # result_df.to_csv("temp.csv", sep=",", index=False)

    return render_template('home.html', type_list=type_list, error_message=error_message, 
    selected_type=selected_type, selected_des=selected_des, target_name=target_name,
    result_header=result_df.columns.values, tables=[result_df.to_html(classes='data')],
    graph_url="./static/images/graph.png")


if __name__ == "__main__":
    app.run
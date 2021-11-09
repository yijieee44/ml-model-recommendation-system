from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd

from collections import Counter
from src.model.pycaret_compare import reg_compare, cla_compare

app = Flask(__name__)
type_list = ["Regression", "Classification"]
result = ""

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html', type_list=type_list, result=result, 
    selected_type=type_list[0], selected_des="", target_name="", notBalance=False)


@app.route('/', methods=['POST'])
def submit():
    selected_type = request.form.get('selected_type')
    target_name = request.form.get('target_name')
    notBalance = False
    suitable = ""
    if 'file' not in request.files:
        result = "## Error: No file uploaded"
        return render_template('home.html', type_list=type_list, result=result, selected_type=selected_type, selected_des="", target_name=target_name, notBalance=False, suitable="")

    file = request.files['file']
    
    if file.filename == '':
        result = "## Error: No file selected"
        selected_des = ""

    if file:
        # generate result in string form here
        selected_des = file.filename
        selected_des = selected_des + " -- " + selected_type
    
        df = pd.read_csv(file.stream)
        df = clean(df)


        if selected_type == "Regression":
            best_model, result_df, suitable = reg_compare(df, target_name)
        elif selected_type == "Classification":
            target_column = df[target_name]
            count = dict(Counter(target_column))
            average =  len(target_column)/len(count.keys())
            for x in count:
                if count[x] < average/2:
                    notBalance = True
            best_model, result_df, suitable = cla_compare(df, target_name)
        result = result_df.to_string()
    
    return render_template('home.html', type_list=type_list, result=result, selected_type=selected_type, selected_des=selected_des, target_name=target_name, notBalance=notBalance, suitable=suitable)

def clean(df):
    df = df[~df[' Income '].isnull()]
    df[' Income '] = df[' Income '].apply(lambda x: x[1:].replace(',', '').split('.')[0])
    df[' Income '] = df[' Income '].astype('int32')

    df['Education'] = df['Education'].astype('category')
    df['Marital_status'] = df['Marital_Status'].astype('category')
    df['Country'] = df['Country'].astype('category')

    df.drop(['ID', 'Dt_Customer'], inplace=True, axis=1)
    return df


if __name__ == "__main__":
    app.run
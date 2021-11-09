from flask import Flask, render_template, request
import os

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


if __name__ == "__main__":
    app.run
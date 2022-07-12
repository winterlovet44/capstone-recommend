import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from utils.util import load_model
from recommend.contentbase import ContentBased
# from recommend.als import AlternatingLeastSquares

from utils.app_utils import load_data, get_res_result


app = Flask(__name__)


# Global variable
# load data
engine = create_engine("sqlite:///metadata")
df = load_data()
history = load_data("data/rating.csv")

# load model
als = load_model("models/als.pkl")
cb = load_model("models/cb.pkl")
choices = ['als', 'cb']


# def select_model(model_name):
#     if model_name == 'als':
#         return als
#     elif model_name == 'cb':
#         return cb


def select_model(model_name):
    """
    Return the model based on model name.

    Parameters
    ----------
    model_name: str
        name of model, only accept 'als' or 'cb'
    """
    if model_name == 'als':
        return als
    elif model_name == 'cb':
        return cb


# def cb_recommend(input_id)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/home')
def index():
    # render web page with plotly graphs
    return render_template('home.html')


# web page that handles user query and displays model results
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    # get model name from user selected
    model_name = request.args.get("model", "als")
    runner = select_model(model_name)
    # save user input in query
    query = request.args.get('inputid', '')
    query = int(query)
    # use model to predict classification for query
    result = get_res_result(model=runner, dataset=history, user_id=query)

    # render the recommend.html Please see that file.
    return render_template(
        'recommend.html',
        data=result,
        # state=state
    )


# @app.route('/get_data', methods=['POST'])
# def get_data():
#     if request.method == 'POST':
#         print(request.form)
#     print(request.form.get('select1'))
#     return render_template("home.html")


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
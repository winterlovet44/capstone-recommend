# Recommendation Web-app

### Table of Contents

1. [Project Description](#motivation)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [File Descriptions](#files)
5. [Instructions](#results)



## Project Description<a name="motivation"></a>

In this project, I build a application about movie recommender system with Movielens 1M dataset.
For more information about the dataset, please check [here](https://grouplens.org/datasets/movielens/1m/).
I use Alternating Least Square which has been implemented in [implicit](https://github.com/benfred/implicit) for user recommendation and a Content-based module for related item recommendation.
I also implement a simple web-app for perform recommend for an user or an item (model serving).

This project contains four steps of ML pipeline:

1. ETL: Clean data and save cleaned data to file and database.
2. Feature engineering: Transform feature to meet model fitting.
3. Modelling: Build a Machine learning pipeline to feature engineering and train ML model.
4. Model serving: Build Flask web app to predict user's input query.



## Dependencies <a name="dependencies"></a>

To run code in this project, your enviroments need:
1. [Pandas](https://pandas.pydata.org/)
2. [Numpy](https://numpy.org/)
3. [Scikit-learn](https://scikit-learn.org/stable/)
4. [implicit](https://github.com/benfred/implicit)
5. [SQLAlchemy](https://sqlalchemy.org/)
6. [Flask](https://flask.palletsprojects.com/)
7. [Click](https://click.palletsprojects.com/en/8.1.x/)


## Installation <a name="installation"></a>

The code was implemented in Python 3.9. All necessary package was contained in `requirements.txt` file.

For quick installation:
```sh
pip install -r requirements.txt
```


## File Descriptions <a name="files"></a>

```bash
├───data  # Contains data after cleaned
├───models  # Contains pickle file, it will be generate after run model
├───notebooks  # Contains notebook about explore model and serving model
├───raw  # Raw data of Movielens 1M
├───recommend  # Contains some module for recommendation
├───static  # Flask static folder, contains image and css style
│   ├───images
│   └───styles
├───templates  # Flask html folder, use for render
└───utils  # Contains some helper function for project
```

## Instructions<a name="results"></a>

1. ETL pipeline

We need to pre-processing for user's dataset and item's dataset.
To run ETL pipeline for clean user dataset, run the code below:

```bash
python movielens_rating_etl.py
```

To run ETL pipeline for clean item's dataset:

```bash
python movielens_meta_etl.py
```

2. To build and train model

We have 2 model is ALS and ContentBased. 
To run ALS model

```bash
python als.py
```

To run ContentBased model

```bash
python cb.py
```

3. Run web app

Run the code below to start the web app at localhost

```bash
python run.py
```

And go to [http://localhost:3000](http://localhost:3000/) to see the web app



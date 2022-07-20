# Udacity Capstone project - Recommender system

### Motivation
In this project, i will build a recommender system web-app to make recommendation for user. I recommend for user the items they may like and a recommender system to find out which item is similar to this item.
### Table of Contents

1. [Project Description](#motivation)
2. [Installation](#installation)
3. [EDA](#EDA)
4. [Methodology](#method)
5. [Results](#result)
6. [Instructions](#instruc)


## Project Description<a name="motivation"></a>

In this project, I build a application about movie recommender system with Movielens 1M dataset.
For more information about the dataset, please check [here](https://grouplens.org/datasets/movielens/1m/).
I use Alternating Least Square which has been implemented in [implicit](https://github.com/benfred/implicit) for user recommendation and a Content-based module for related item recommendation.
I also implement a simple web-app to perform recommendation for the user or the item (model serving).

This project contains four steps of ML pipeline:

1. ETL: Clean data and save cleaned data to file and database.
2. Feature engineering: Transform feature to meet model fitting.
3. Modelling: Build a Machine learning pipeline to feature engineering and train ML model.
4. Model serving: Build Flask web app to predict user's input query.



## Installation <a name="installation"></a>

The code was implemented in Python 3.9. All necessary package was contained in `requirements.txt` file.

For quick installation:
```sh
pip install -r requirements.txt
```


## EDA <a name="eda"></a>

The Movielens 1M dataset contains information about history of user and movie's profile.
To see the EDA of Movielens 1M, please go to this [notebook](https://github.com/winterlovet44/capstone-recommend/blob/main/notebooks/movielens_ETL_and_EDA.ipynb)


## Methodology <a name="method"></a>

### **Model**

1. Alternating Leasts Squares (ALS): An approach of matrix factorization. this model try to decompose rating matrix into two factos matrix
2. Content-based (CB): A content based approach use cosine similarity to find most similar item.

With ALS model, i use implementation from [implicit](https://github.com/benfred/implicit) for better performance. With CB model, i implement my own and try to combine multiple of data type.
My CB implementation can handle multiple of content with data type can be list, category or text. Final similarity of pair items is average of all features input.
Code of this implementation you can find [here](https://github.com/winterlovet44/capstone-recommend/blob/main/recommend/contentbase.py)


### **Metrics**

In this project, i only implement evaluation for ALS. To evaluate ALS, I use 3 metrics: RMSE, MAP@k and P@k.


1. RMSE (Root Mean Squares Error): the differences between predicted rating and true rating.
2. P@k (Precision at k): Precision of recommendation with top k result.
3. MAP@k (Mean average precision at k): Mean of P@k with all users.


## Results<a name="result"></a>

The result of ALS for Movielens 1M. You can view detail at [here](https://github.com/winterlovet44/capstone-recommend/blob/main/notebooks/ALS-experiment.ipynb)

| Factors | RMSE     | MAP@k     | P@k       |
|---------|----------|-----------|-----------|
| 10      | 3.21     | 0.10      | 0.206     |
| 30      | 3.18     | 0.122     | 0.244     |
| **50**  | **3.18** | **0.128** | **0.246** |
| 100     | 3.214    | 0.121     | 0.234     |
 | 300     | 3.42     | 0.087     | 0.171     |
| 1000    | 3.67     | 0.0364    | 0.0733    |



## Instructions<a name="instruc"></a>

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



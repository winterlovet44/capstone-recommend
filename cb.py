import os
import click
import pandas as pd

from recommend.contentbase import ContentBased
from utils.util import save_model
from utils.variables import METADATA, CB_MODEL_PATH

# import numpy as np
# from sqlalchemy import create_engine

# Set `num threads` = 1 for BLAS package
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

CB_CFG = {
    "id_col": "MovieID",
    "text_cols": ["Title"],
    "categorical_cols": ['release_year'],
    "list_cols": ["Genres"]
}


def load_meta(path):
    ftype = path.split(".")[-1]
    if ftype == "json":
        return pd.read_json(path, orient="records")
    elif ftype == 'csv':
        return pd.read_csv(path)
    elif ftype == 'parquet':
        return pd.read_parquet(path)
    else:
        raise TypeError(f"Unknown {ftype} file, please input csv, json or parquet file")


def content_based_train(data, config=CB_CFG):
    model = ContentBased()
    model.fit(data, config)
    return model


def save_model(model, path):
    if hasattr(model, "save"):
        model.save(path)
    else:
        save_model(model, path)


@click.command()
@click.option('--input_path', default=METADATA, help='Filepath contains movielens 1M dataset')
@click.option('--model_path', default=CB_MODEL_PATH, help='Output path to write cleaned metadata')
# @click.option('--model_name', default="cb", help='Output path to write cleaned metadata')
def run(input_path, model_path):
    print("Load metadata...")
    df = load_meta(input_path)
    print("Initialize model and train...")
    model = content_based_train(df)
    print("Save model...")
    print(f"Save model to {model_path}")
    save_model(model, model_path)
    print("done!!!")


if __name__ == '__main__':
    run()

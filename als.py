import os
import click

from recommend.dataset import Dataset
import pandas as pd

from implicit.als import AlternatingLeastSquares
from utils.util import save_model
from utils.variables import RATING, ALS_MODEL_PATH


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def load_data(path):
    """Load rating data to Dataset.

    Parameters
    ----------
    path: str
        Path string to rating data file

    Returns
    -------
    Dataset contain data from rating.csv
    """
    df = pd.read_csv(path, index_col=0)
    return Dataset(df, user="UserID", item="MovieID", rating="Rating")


@click.command()
@click.option('--input_path', default=RATING, help='Filepath contains movielens 1M dataset')
@click.option('--model_path', default=ALS_MODEL_PATH, help='Output path to write cleaned metadata')
def run(input_path, model_path):
    """Main function to train and save ALS model."""
    print(f"Load data from {input_path}\n")
    ds = load_data(input_path)
    print("Initialize model...\n")
    model = AlternatingLeastSquares(num_threads=1, random_state=1)
    print("Fit data to model")
    model.fit(ds.get_csr())
    print("Done train model")
    print(f"User embedding shape: {model.user_factors.shape}")
    print(f"User embedding shape: {model.item_factors.shape}")
    print("Save model")
    save_model(model, model_path)
    print("Done!!!")


if __name__ == '__main__':
    run()

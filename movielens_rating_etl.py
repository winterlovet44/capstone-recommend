import click
import pandas as pd
from utils.util import ensure_type_path

COLUMNS = ['UserID', "MovieID", "Rating"]


def load_data(path):
    """Function for load raw data of movielens 1M.

    Parameters
    ----------
    path: str
        Path contains raw movielens 1M rating data

    Returns
    -------
    Dataframe contains rating data of movielens 1M"""
    df = pd.read_csv(path,
                     delimiter="::",
                     names=['UserID', "MovieID", "Rating", "Timestamp"],
                     engine='python'
                     )
    return df
#
#
# def ensure_csv_path(path):
#     if path.split(".")[-1] != "csv":
#         path += "rating.csv"
#     return path


@click.command()
@click.option('--input_path', help='Filepath contains user\
                                rating of movielens 1M dataset')
@click.option('--write_to', default="data/rating.csv", help='Output path\
                                to write cleaned metadata')
def run(input_path, write_to):
    """Main function to run ETL of rating data."""
    print("Load raw data...\n")
    df = load_data(input_path)
    df = df[COLUMNS]
    write_to = ensure_type_path(write_to, "csv")
    print(f"Write metadata to {write_to}\n")
    df.to_csv(write_to)
    print("Done!!!")
    return


if __name__ == '__main__':
    run()

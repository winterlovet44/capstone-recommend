import pickle
import pandas as pd
import numpy as np
from recommend.dataset import Dataset
from sqlalchemy import create_engine
from recommend.contentbase import ContentBased
from utils.util import get_connection_to_meta
from utils.variables import RATING, TABLE


engine = get_connection_to_meta()
# RATING_PATH = "data/rating.csv"


def load_data(path=RATING):
    """Load rating dataset to Dataset module.

    Parameters
    ----------
    path: str, default: data/rating.csv
        Path of rating data saved

    Returns
    -------
    Dataset: Dataset instance contain rating data
    """
    df = pd.read_csv(path, index_col=0)
    return Dataset(df, user="UserID", item="MovieID", rating="Rating")


def get_user_history(user_id, dataset):
    """Return history rating of an user.

    Parameters
    ----------
    user_id: int
        Id of user
    dataset: Dataset
        Dataset instance, contains rating data

    Returns
    -------
    User's history
    """
    if user_id not in dataset.user_list:
        raise ValueError(f"User id {user_id} not in dataset")
    return dataset.get_user_history(user_id)


# def read_data(query, conn):
#     """Wrapper of pandas.read_sql_query."""
#     return pd.read_sql_query(query, conn)


def get_movie_information(iid, sql_engine=engine, table=TABLE):
    """
    Load metadata in SQLite database by using pandas.read_sql_query.

    Parameter:
    iid: int
        Item id for load metadata
    sql_engine: SQL connection
        connection to db
    table: str
        Table name of metadata, default: movielens
    return: pd.DataFrame
        Dataframe contains metadata of input item.
    """
    query = f"""SELECT * FROM {table} WHERE MovieID = {iid}"""
    return pd.read_sql_query(query, sql_engine)


def get_metadata(*items):
    """Return Datafrane contains metadata of list items.

    Parameters
    ----------
    items: array-like
        List of item

    Returns
    -------
    Dataframe contains information of list item.
    """
    res = []
    for item in items[0]:
        res.append(get_movie_information(item))
    return pd.concat(res, ignore_index=True)


def parse_cb_result(model, iid, topk=10):
    """Function help parse result of Content-based model.

    Parameters
    ----------
    model: ContentBased
        Content based model
    iid: int
        Item id will be recommended
    topk: int, default: 10
        Top related item

    Returns
    -------
    Dataframe of list item has the highest correlation with input item.
    """
    res_id, res_score = model.recommend(iid, topk)
    df = get_metadata(res_id)
    df['score'] = res_score
    return df


def get_res_result(model, user_id, dataset, topk=10):
    """Function help parse result of ALS model.

    Parameters
    ----------
    model: ContentBased
        ALS model
    user_id: int
        User id will be recommended
    dataset: Dataset
        Dataset instance contain rating data
    topk: int, default: 10
        Top related item

    Returns
    -------
    Dataframe of list item will be recommended for target user.
    """
    if isinstance(model, ContentBased):
        return parse_cb_result(model, user_id, topk)
    if user_id not in dataset.user:
        raise ValueError(f"User id {user_id} not in dataset")
    idx = np.where(dataset.user_list == user_id)[0][0]
    item_id, score = model.recommend(idx, dataset.get_csr()[idx], N=topk)
    df = get_metadata(item_id)
    df['score'] = score
    return df

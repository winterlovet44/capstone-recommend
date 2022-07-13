import pickle
from recommend import contentbase # noqa
from utils import variables
from sqlalchemy import create_engine


def ensure_type_path(path, typefile='csv'):
    """
    Function for check input file path is correct.
    If this file path is not correct with expected type file
    function will be added typefile at the end of path

    Parameter
    path: str
        String file path will be checked

    typefile: str
        Type of file

    return:
    path: str
        File path after check
    """
    if path.split(".")[-1] != typefile:
        path += "." + typefile
    return path
#
#
# def save_model(model, path):
#     with open(path, "wb") as f:
#         pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
#         f.close()
#     return


def save_model(model, path):
    """
    Function for Serialization model.
    Using for save model as a pickle file.

    Parameter
    model: object
        Model will be saved
    path: str
        File path for save model
    return: None
    """
    if hasattr(model, "save"):
        model.save(path)
    else:
        with open(path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
    return


def load_model(path):
    """
    Function for Deserialization model.
    Using for load model from a pickle file.

    Parameter
    path: str
        File path for load model
    return
    model: object
        Model will be loaded
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
        f.close()
    return model


def get_connection_to_meta():
    return create_engine(f"sqlite:///{variables.DATABASE}")

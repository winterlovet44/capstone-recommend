
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseEncoder:
    """Base class contains fit method for another encoder."""

    def _fit(self, data):
        pass

    def fit(self, data, return_type="sparse"):
        """Perform One-hot Encode method based on data type.

        This function will be call _fit function to transform data to
        One-hot matrix based on data type.

        Parameters
        ----------
        data: pd.DataFrame
            Dataframe will be transformed
        return_type: str, default: sparse
            Matrix type will be returned

        Returns
        -------
        One-hot matrix with type based on return_type parameter.
        """
        array = self._fit(data)
        if return_type == 'sparse':
            return csr_matrix(array)
        elif return_type == 'dataframe':
            cols = self.classes
            idx = self.index
            return pd.DataFrame(array, columns=cols, index=idx)
        else:
            return array

    def __call__(self, data, return_type='sparse'):
        return self.fit(data, return_type)


class TfidfEncoder(BaseEncoder):
    """Encode text data using TfIDF method."""

    def __init__(self, **kwargs):
        self.engine = TfidfVectorizer(max_features=100)

    def _fit(self, data):
        arr = self.engine.fit_transform(data)
        self.classes = [f"col_{x}" for x in range(self.engine.max_features)]
        if not hasattr(data, "index"):
            self.index = list(range(len(data)))
        else:
            self.index = data.index.copy()
        return arr


class CategoricalEncoder(BaseEncoder):
    """Perform one-hot encoder for categorical data using LabelBinarizer method."""

    def __init__(self, **kwargs):
        self.engine = LabelBinarizer()

    def _fit(self, data):
        arr = self.engine.fit_transform(data)
        self.classes = self.engine.classes_
        if not hasattr(data, "index"):
            self.index = list(range(len(data)))
        else:
            self.index = data.index.copy()
        return arr


class ListEncoder(BaseEncoder):
    """Perform one-hot encoder for nested list categorical data using
    MultiLabelBinarizer method.

    """

    def __init__(self, **kwargs):
        self.engine = MultiLabelBinarizer()

    def _fit(self, data):
        arr = self.engine.fit_transform(data)
        self.classes = self.engine.classes_
        if not hasattr(data, "index"):
            self.index = list(range(len(data)))
        else:
            self.index = data.index.copy()
        return arr


class ContentBased:
    """
    Recommender system based on content of item.

    This class contains method for build matrix and compute similarity
    of each item by its profile.

    Model can handle text, category or nested-category data and transform
    it into one-hot matrix.

    Class using sklearn.metrics.pairwise.cosine_similarity to compute similarity
    between items.
    """

    def __init__(self):
        self.n_feartures = 0
        self.list_features = []
        self.cate_features = []
        self.text_features = []
        # self.similarity = similarity
        self.simi_matrix = np.array([])
        self.categorical_encoder = CategoricalEncoder()
        self.text_encoder = TfidfEncoder()
        self.list_encoder = ListEncoder()
        self.item_index_map = {}
        self.item_id = []

    @staticmethod
    def cosine_similarity(arr):
        """Wrapper function of sklearn.metrics.pairwise.cosine_similarity."""
        return cosine_similarity(arr)

    def recommend(self, iid, k=10):
        """
        Find top N most similar item for input item.

        Function will return top N most similar item and its score.

        Parameters
        ----------
        iid: int
            Input item id for recommendation
        k: int, default: 10
            Top N item most similar

        Returns
        -------
        Tuple of (item id, score) has the highest correlation with input item.
        """
        # Check if model not fitted
        if self.simi_matrix.size == 0:
            raise NotImplementedError(f"Model not fit yet")

        idx = self.item_index_map.get(iid)
        if idx is None:
            return {}
        item_vector = self.simi_matrix[idx]
        top = np.argsort(item_vector)[::-1][:k]
        item_rec = [self.item_id[x] for x in top]
        return (np.array(item_rec), item_vector[top])

    def fit(self, data, config):
        """
        Fit data and it's config to compute similarity.

        The similarity matrix is average of all similarity matrix
        from all features, and it's saved as simi_matrix

        Parameters
        ----------
        data: pd.DataFrame
            Dataframe contains profile of item.
        config: dict
            Dictionary contains info about data.
            It must have struct
            {
                "id_col": "id",
                "text_cols": ["x"],
                "categorical_cols": ["y"],
                "list_cols": ["z"]
            }

        Returns
        -------
        None
        """
        self.item_id = data[config['id_col']]
        self.item_index_map = {v: k for k, v in self.item_id.to_dict().items()}
        self.text_features = config['text_cols']
        self.cate_features = config['categorical_cols']
        self.list_features = config['list_cols']
        self.n_feartures = len(self.text_features) + \
                           len(self.cate_features) + \
                           len(self.list_features)

        self.simi_matrix = np.zeros((len(self.item_id), len(self.item_id)))

        if self.text_features:
            for cols in self.text_features:
                print(f"Compute similarity of feature {cols}")
                s_arr = self.cosine_similarity(self.text_encoder(data[cols]))
                self.simi_matrix += s_arr
        if self.cate_features:
            for cols in self.cate_features:
                print(f"Compute similarity of feature {cols}")
                s_arr = self.cosine_similarity(self.categorical_encoder(data[cols]))
                self.simi_matrix += s_arr
        if self.list_features:
            for cols in self.list_features:
                print(f"Compute similarity of feature {cols}")
                s_arr = self.cosine_similarity(self.list_encoder(data[cols]))
                self.simi_matrix += s_arr

        self.simi_matrix /= self.n_feartures
        print(f"Done train model, total {self.n_feartures} has trained")

    def save(self, path):
        """Dump model to a pickle file for future using.

        Parameters
        ----------
        path: str
            Path string for model saved

        Returns
        -------
        None
        """
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

    # @classmethod
    # def load(cls, path):
    #     with open(path, "rb") as f:
    #         model = pickle.load(f)
    #     return model
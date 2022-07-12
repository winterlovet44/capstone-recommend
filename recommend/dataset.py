
import numpy as np
import pandas as pd
from scipy import sparse
# from sklearn.model_selection import train_test_split


class Dataset:
    """A dataset contains rating data of movielens.

    This class contains method for transform data from
    Dataframe to sparse matrix and access to history of users.
    """
    
    def __init__(self,
                 df,
                 user='user',
                 item='item',
                 rating='rating',
                 dtype=np.float32,
                 split_frac=0.05
                 ):

        self._user = user
        self._item = item
        self._rating = rating
        self.df = df
        self.dtype = dtype
        self.user_list = np.array([])
        self.item_list = np.array([])
        self.df['user'], self.user_list = self.get_factorizer('user')
        self.df['item'], self.item_list = self.get_factorizer('item')
    
    @property
    def user(self):
        return self.df[self._user]
    
    @property
    def item(self):
        return self.df[self._item]
    
    @property
    def rating(self):
        return self.df[self._rating]
        
    def get_factorizer(self, attr='user'):
        """Encode user id by integer number."""
        if attr not in ['user', 'item']:
            raise ValueError(f"Unknown attribute {attr}, only user, item was accepted")
        data = getattr(self, attr)
        return pd.factorize(data)
    
    def transform_to_matrix(self):
        """Function to transform pairwise to matrix.

        The data in rating data save as format |user_id|item_id|rating|
        This function will be transformed it formats to scipy.sparse.coo_matrix
        """
        n_user: int = len(self.user_list)
        n_item: int = len(self.item_list)
        return sparse.coo_matrix(
            (self.rating, (self.df.user, self.df.item)),
            shape=(n_user, n_item),
            dtype=self.dtype
        )
    
    def get_csr(self):
        """Get rating data as scipy.sparse.csr_matrix."""
        return self.transform_to_matrix().tocsr()
    
    def get_csc(self):
        """Get rating data as scipy.sparse.csc_matrix."""
        return self.transform_to_matrix().tocsc()
    
    def get_user_history(self, uid):
        """Get history of certain user.

        Parameters
        ----------
        uid: int
            Id of user will be retrieved

        Returns
        -------
        User id and their history
        """
        res = {"user id": uid}
        history = self.get_csr()
        user_idx = np.where(self.user_list == uid)[0][0]
        list_idx_item = history[user_idx].indices
        item_id = self.item_list[list_idx_item]
        res['item id watched'] = item_id
        return res
    
    def get_item_history(self, iid):
        """Get list user has been watched this item.

        Parameters
        ----------
        iid: int
            Id of item will be retrieved

        Returns
        -------
        Item id and list of user has been watched it.
        """
        res = {"Item id": iid}
        history = self.get_csc()
        item_idx = np.where(self.item_list == iid)[0][0]
        list_idx_user = history[:, item_idx].indices
        user_id = self.user_list[list_idx_user]
        res['Users has been watch this item'] = user_id
        return res

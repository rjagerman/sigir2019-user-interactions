import numba
import numpy as np
import json
from ltrpy.evaluation import ndcg
from ltrpy.ranker import inorder


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@numba.njit
def numba_seed(seed):
    np.random.seed(seed)


def rng_seed(seed):
    numba_seed(seed)
    return np.random.RandomState(seed)


@numba.njit(nogil=True)
def score(x, w):
    return np.dot(x, w)


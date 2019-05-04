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


class StrRepCache:
    def __init__(self, zero_based):
        self._str_rep_cache = {}
        self.zero_based = zero_based

    def get_str_rep(self, qid, x, i):
        if (qid, i) not in self._str_rep_cache.keys():
            l = []
            for j in np.argwhere(x[i, :] != 0.0).flatten():
                l.append(f"{j + self.zero_based}:{x[i, j]}")
            self._str_rep_cache[(qid,i)] = ' '.join(l)
        return self._str_rep_cache[(qid, i)]


@numba.njit
def numba_seed(seed):
    np.random.seed(seed)


def rng_seed(seed):
    numba_seed(seed)
    return np.random.RandomState(seed)


@numba.njit(nogil=True)
def score(x, w):
    return np.dot(x, w)


@numba.njit(nogil=True, parallel=True)
def eval_dataset(data, w):
    ndcg_at_10 = 0.0
    rrr = 0
    rrr_n = 0
    for i in numba.prange(len(data)):
        x, y, qid = data[i]
        s = score(x, w)
        r = inorder(s)
        ndcg_at_10 += ndcg(r, y, exp=True)[:10][-1]
        
        for j in range(y.shape[0]):
            if y[r[j]] >= 3:
                rrr += j
                rrr_n += 1
    results = dict()
    results["ndcg@10"] = ndcg_at_10 / len(data)
    results["arrr"] = rrr / rrr_n
    return results


def model_to_svmrank_str(w, zero_based=False):
    out = "SVM-light Version V6.20\n"
    out += "0 # kernel type\n"
    out += "3 # kernel parameter -d\n"
    out += "1 # kernel parameter -g\n"
    out += "1 # kernel parameter -s\n"
    out += "1 # kernel parameter -r\n"
    out += "empty# kernel parameter -u\n"
    out += "700 # highest feature index\n"
    out += "45 # number of training documents\n"
    out += "2 # number of support vectors plus 1\n"
    out += "0 # threshold b, each following line is a SV (starting with alpha*y)\n"
    out += "1 "
    incr = 0 if zero_based else 1
    for i in range(w.shape[0]):
        out += f"{i + incr}:{w[i]} "
    out += "#"
    return out


# @numba.jitclass([
#     ('_values', numba.typeof(DynamicArrayF64(16))),
#     ('_starts', numba.typeof(DynamicArrayI64(16))),
#     ('_ends', numba.typeof(DynamicArrayI64(16)))
# ])
# class ListOfVectorsF64:
#     def __init__(self):
#         self._values = DynamicArrayF64(16)
#         self._starts = DynamicArrayI64(16)
#         self._ends = DynamicArrayI64(16)
    
#     def add(self, values):
#         self._starts.append(self._values.size)
#         self._values.extend(values)
#         self._ends.append(self._values.size)
    
#     def get(self, index):
#         if index >= self.size or -index > self.size:
#             raise IndexError
#         start = self._starts.array[index]
#         end = self._ends.array[index]
#         return self._values.array[start:end]
        
#     @property
#     def size(self):
#         return self._starts.size


# @numba.jitclass([
#     ('_values', numba.typeof(DynamicArrayI64(16))),
#     ('_starts', numba.typeof(DynamicArrayI64(16))),
#     ('_ends', numba.typeof(DynamicArrayI64(16)))
# ])
# class ListOfVectorsI64:
#     def __init__(self):
#         self._values = DynamicArrayI64(16)
#         self._starts = DynamicArrayI64(16)
#         self._ends = DynamicArrayI64(16)
    
#     def add(self, values):
#         self._starts.append(self._values.size)
#         self._values.extend(values)
#         self._ends.append(self._values.size)
    
#     def get(self, index):
#         #if index >= self.size or -index > self.size:
#         #    raise IndexError
#         start = self._starts.array[index]
#         end = self._ends.array[index]
#         return self._values.array[start:end]
        
#     @property
#     def size(self):
#         return self._starts.size

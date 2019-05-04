import numpy as np
import numba
import json
from argparse import ArgumentParser
from ltrpy.ranker import inorder
from ltrpy.evaluation import ndcg
from sigir2019.util import score


def _ndcg(labels, scores, prng):
    denom = 1./np.log2(np.arange(10)+2.)
    labels = np.array(labels)
    scores = np.array(scores)

    random_i = prng.permutation(np.arange(scores.shape[0]))

    labels = labels[random_i]
    scores = scores[random_i]

    sort_i = np.argsort(scores)
    sorted_labels = labels[sort_i][:-11:-1]
    k = sorted_labels.shape[0]
    nom = 2**sorted_labels-1.
    dcg = np.sum(nom*denom[:k])

    ilabels = np.sort(labels)[:-k-1:-1]
    inom = 2**ilabels-1.
    idcg = np.sum(inom*denom[:k])
    return dcg/max(idcg, 1.)


def evaluate(data, model, prng, filter=True):
    ndcgs = []
    rrrs = []
    for r in range(data.size):
        x, y, qid = data.get(r)
        if not filter or np.sum(y) > 0:
            scores = np.dot(x, model).flatten()
            ndcgs.append(_ndcg(y, scores, prng))
            p = prng.permutation(scores.shape[0])
            r = p[np.argsort(-scores[p])]
            rrr = np.argwhere(y[r] >= 3)
            rrrs.extend(rrr.tolist())

    return {
        "ndcg@10": np.mean(ndcgs),
        "rrr": np.mean(rrrs)
    }


def evaluate2(data, model, filter=True):
    score_ndcg, score_rrr = _evaluate(data, model, filter)
    return {
        "ndcg@10": score_ndcg,
        "rrr": score_rrr
    }


@numba.njit(nogil=True)
def _evaluate(data, model, filter):
    score_ndcg = 0.0
    score_rrr = 0.0
    sum_ndcg = []
    sum_rrr = []
    for i in range(data.size):
        x, y, qid = data.get(i)
        if not filter or np.sum(y) > 0:
            s = score(x, model)
            r = inorder(s)
            q_ndcg = ndcg(r, y, True)[:10][-1]
            q_rrr = np.where(y[r] >= 3)[0]
            sum_ndcg.append(q_ndcg)
            sum_rrr.extend(q_rrr)
    score_ndcg = np.mean(np.array(sum_ndcg))
    score_rrr = np.mean(np.array(sum_rrr))
    return score_ndcg, score_rrr

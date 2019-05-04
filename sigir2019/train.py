import numpy as np
import numba
import logging
from time import time
from sigir2019.util import score
from ltrpy.ranker import inorder
from rulpy.optimization import SGD, Adam
from rulpy.math import hinge, grad_hinge, log_sum_exp, grad_additive_dcg


@numba.njit(nogil=True)
def pl_log_prob(r, s):
    out = 0.0
    for i in range(s.shape[0]):
        out += s[r[i]] - log_sum_exp(s[r[i:]])
    return out


@numba.njit(nogil=True)
def lse_2(e1, e2):
    m = max(e1, e2)
    return m + np.log(np.exp(e1 - m) + np.exp(e2 - m))


@numba.njit(nogil=True)
def regularize(w, l1, l2):
    if l2 > 0.0:
        w = w - l2 * w
    
    if l1 > 0.0:
        w = np.sign(w) * np.maximum(np.zeros(w.shape), np.abs(w) - l1 * np.abs(w))
    
    return w


@numba.njit(nogil=True)
def gradient_pairwise_hinge(x, w, r, c, p_eta):
    final_grad = np.zeros(w.shape)
    s = score(x, w)
    for i in c:
        grad = np.zeros(w.shape)
        propensity = (1.0 / (i + 1)) ** p_eta
        for j in range(r.shape[0]):
            if i != j:
                f_i = x[r[i], :]
                f_j = x[r[j], :]
                s_ij = s[r[i]] - s[r[j]] # score(f_i - f_j, w) #
                g = grad_hinge(s_ij)
                grad += (f_i - f_j) * g
        grad = grad / propensity
        final_grad += grad
    return final_grad


@numba.njit(nogil=True)
def gradient_pairwise_dcg(x, w, r, c, p_eta):
    final_grad = np.zeros(w.shape)
    s = score(x, w)
    for i in c:
        grad = np.zeros(w.shape)
        propensity = (1.0 / (i + 1)) ** p_eta
        h = 1.0
        for j in range(r.shape[0]):
            if i != j:
                f_i = x[r[i], :]
                f_j = x[r[j], :]
                s_ij = s[r[i]] - s[r[j]]
                h += hinge(s_ij)
                g = grad_hinge(s_ij)
                grad += (f_i - f_j) * g
        grad = grad * grad_additive_dcg(h)
        grad = grad / propensity
        final_grad += grad
    return final_grad


@numba.njit(nogil=True)
def gradient_pairwise_exp(x, w, r, c, p_eta):
    grad = np.zeros(w.shape)
    s = score(x, w)
    for i in c:
        for j in range(r.shape[0]):
            if i != j:
                propensity = (1.0 / (i + 1)) ** p_eta
                f_i = x[r[i], :]
                f_j = x[r[j], :]
                lse = lse_2(s[r[i]], s[r[j]])
                g = np.exp(s[r[i]] + s[r[j]] - 2 * lse)
                grad -= (f_i - f_j) * g / propensity
    return grad


gradients = {
    "hinge": gradient_pairwise_hinge,
    "exp": gradient_pairwise_exp,
    "dcg": gradient_pairwise_dcg
}


def train_model(data, indices, rankings, clicks, gradient, epochs, lr,
                p_eta=1.0, l1=0.0, l2=0.0, decay=1.0, seed=None):
    
    # Initialize model and optimizer
    gradient = gradients[gradient]
    w = np.zeros(data.d)
    sgd = SGD(w, lr)

    # Run optimization loop
    logging.info(f"Running optimization ({epochs} epochs, {len(indices)} samples)")
    start = time()
    w = _train_loop(data, indices, rankings, clicks, w, gradient, sgd, epochs,
                    p_eta, l1, l2)
    logging.info(f"Optimization took {time() - start:.1f} seconds")

    # Return model
    return w


@numba.njit(nogil=True)
def _train_loop(data, indices, rankings, clicks, w, gradient, sgd, epochs,
                p_eta, l1, l2):
    for e in range(epochs):
        p = np.random.permutation(len(indices))
        for i in range(len(indices)):
            x, y, qid = data.get(indices[p[i]])
            r = rankings.get(p[i])
            c = clicks.get(p[i])
            g = gradient(x, w, r, c, p_eta)
            w = sgd.update(g)
    return w

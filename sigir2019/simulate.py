import numba
import numpy as np
from sigir2019.util import score
from rulpy.array import GrowingArrayList
from ltrpy.ranker import inorder
from ltrpy.clicks.cascading import perfect_5
from ltrpy.clicks.position import position_binarized_5, near_random_5


behaviors = {
    "position": lambda cutoff, eta, eps1, eps2: position_binarized_5(cutoff, eta, eps1, eps2),
    "perfect": lambda cutoff, eta, eps1, eps2: perfect_5(cutoff),
    "random": lambda cutoff, eta, eps1, eps2: near_random_5(cutoff, eta)
}


def run_click_simulation(dataset, indices, behavior, baselines):
    rankings = GrowingArrayList(dtype=numba.int64)
    clicks = GrowingArrayList(dtype=numba.int64)
    _run_click_simulation(dataset, indices, behavior, baselines, rankings, clicks)
    return rankings, clicks


@numba.njit(nogil=True)
def _run_click_simulation(dataset, indices, behavior, baselines, rankings, clicks):
    #rankings = ListOfVectorsI64()
    #clicks = ListOfVectorsI64()
    
    baseline = 0
    w = np.zeros(0, dtype=np.float64)

    for i in range(len(indices)):
        if baseline < len(baselines):
            if i == baselines[baseline][0]:
                w = baselines[baseline][1]
                baseline += 1
        
        x, y, qid = dataset.get(indices[i])
        s = score(x, w)
        r = inorder(s)
        c = behavior.simulate(r, y)

        rankings.append(r)
        clicks.append(np.where(c > 0)[0])
    #return rankings, clicks

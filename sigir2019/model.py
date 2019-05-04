import numpy as np


def load_svmrank_model(path, zero_based=True):
    with open(path, 'rt') as f:
        lines = f.readlines()
        size = int(lines[7].split(" ")[0])
        if not zero_based:
            size -= 1
        w = np.zeros(size)
        features = lines[-1].split(" ")
        for f in features[1:-1]:
            k, v = f.split(":")
            k = int(k)
            if not zero_based:
                k -= 1
            w[k] = float(v)
    return w

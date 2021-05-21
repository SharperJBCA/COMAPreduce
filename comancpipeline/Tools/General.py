import numpy as np

def read_2bit(feature):
    """
    For a given 2 bit feature, read all features
    """

    top_feature = np.floor(np.log(feature)/np.log(2))

    features = [top_feature]
    while True:
        fs = feature - np.sum([2**f for f in features])

        if fs <= 0:
            break

        next_feature = np.floor(np.log(fs)/np.log(2))
        features += [next_feature]

    return [int(f) for f in features]


def get_obsfeatures(features):

    u_features = np.unique(features)

    features = []
    for ufeature in u_features:
        features += read_2bit(ufeature)

    return np.unqiue(features)

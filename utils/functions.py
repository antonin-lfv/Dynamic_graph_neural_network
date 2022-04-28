from utils.const import *

def distance_neurons(x: list, y: list) -> float:  # distance between 2 neurons
    return fastdist.euclidean(np.array(x), np.array(y))

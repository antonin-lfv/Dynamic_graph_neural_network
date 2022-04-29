from utils.const import *

def distance_neurons(x: list, y: list) -> float:
    """ Distance entre 2 neurones
    :param x: vecteur du premier neurone de taille l
    :param y: vecteur du deuxieme neurone de taille l
    """
    return fastdist.euclidean(np.array(x), np.array(y))
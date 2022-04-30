import numpy as np

from utils.const import *

cosinus = {
    1: 1*np.cos(np.linspace(0, 5, ConstGraph.INPUT_SIZE)),
    2: 2*np.cos(np.linspace(0, 5, ConstGraph.INPUT_SIZE)),
    3: 3*np.cos(np.linspace(0, 5, ConstGraph.INPUT_SIZE)),
    4: 4*np.cos(np.linspace(0, 5, ConstGraph.INPUT_SIZE)),
    5: 5*np.cos(np.linspace(0, 5, ConstGraph.INPUT_SIZE)),
}

sqrt = {
    1: 4 * np.sqrt(np.linspace(0, 5, ConstGraph.INPUT_SIZE)),
    2: 5 * np.sqrt(np.linspace(0, 5, ConstGraph.INPUT_SIZE)),
    3: 6 * np.sqrt(np.linspace(0, 5, ConstGraph.INPUT_SIZE)),
    4: 7 * np.sqrt(np.linspace(0, 5, ConstGraph.INPUT_SIZE)),
    5: 8 * np.sqrt(np.linspace(0, 5, ConstGraph.INPUT_SIZE)),
}

def distance_neurons(x: list, y: list) -> float:
    """ Distance euclidienne entre 2 vecteurs de neurones
    :param x: vecteur du premier neurone de taille l
    :param y: vecteur du deuxieme neurone de taille l
    """
    return round(fastdist.euclidean(np.array(x), np.array(y)), 3)

def get_foyer(graph, neuron):
    """Retourne l'index, la distance et le label du foyer d'un neurone d'entrée
    :param graph:
    :param neuron: le neurone d'entrée
    """
    if len(graph.neurons) != 0:
        distance_foyer, foyer = np.inf, graph.neurons[list(graph.neurons.keys())[0]]
        for n in graph.neurons.keys():
            if distance_neurons(neuron.vecteur, graph.neurons[n].vecteur) < distance_foyer:
                foyer = graph.neurons[n]
        return foyer
    else:
        raise ValueError("Le graphique ne contient aucun neurone")


def get_intersections_2circle(x0: float, y0: float, r0: float, x1: float, y1: float, r1: float):
    # circle 1: (x0, y0), rayon r0
    # circle 2: (x1, y1), rayon r1

    d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # non-intersecting
    if d > r0 + r1:
        return None
    # One circle within other
    if d < abs(r0 - r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(r0 ** 2 - a ** 2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d

        # x4 = x2 - h * (y1 - y0) / d
        # y4 = y2 + h * (x1 - x0) / d

        # return [x3, y3, x4, y4]
        # On choisi arbitrairement un des 2 pts d'intersection
        return x3, y3
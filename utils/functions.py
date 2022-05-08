from utils.const import *


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
            if (d := distance_neurons(neuron.vecteur, graph.neurons[n].vecteur)) < distance_foyer:
                foyer = graph.neurons[n]
                distance_foyer = d
        return foyer
    else:
        raise ValueError("Le graphique ne contient aucun neurone")


def solve_inter_circles(centres_x, centres_y, rayons):
    """Get one intersection of n circles
    :param centres_x: liste des abscisses des n centres des cercles
    :param centres_y: liste des ordonnées des n centres des cercles
    :param rayons: liste des rayons des n cercles
    """

    def func(x):
        return [(x[0] - cx) ** 2 + (x[1] - cy) ** 2 - d ** 2 for cx, cy, d in zip(centres_x, centres_y, rayons)]

    root = fsolve(func, np.array([1] * len(centres_x)), maxfev=500)
    return root[0], root[1]
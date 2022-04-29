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
    return fastdist.euclidean(np.array(x), np.array(y))

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
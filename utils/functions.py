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
    1: 1 * np.sqrt(np.linspace(0, 5, ConstGraph.INPUT_SIZE)),
    2: 2 * np.sqrt(np.linspace(0, 5, ConstGraph.INPUT_SIZE)),
    3: 3 * np.sqrt(np.linspace(0, 5, ConstGraph.INPUT_SIZE)),
    4: 4 * np.sqrt(np.linspace(0, 5, ConstGraph.INPUT_SIZE)),
    5: 5 * np.sqrt(np.linspace(0, 5, ConstGraph.INPUT_SIZE)),
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
        index_foyer, distance_foyer, label_foyer = None, np.inf, None
        for n in graph.neurons.keys():
            if distance_neurons(neuron.vecteur, graph.neurons[n].vecteur) < distance_foyer:
                index_foyer = graph.neurons[n].index
                distance_foyer = distance_neurons(neuron.vecteur, graph.neurons[n].vecteur)
                label_foyer = graph.neurons[n].label
        return index_foyer, distance_foyer, label_foyer
    else:
        raise ValueError("Le graphique ne contient aucun neurone")
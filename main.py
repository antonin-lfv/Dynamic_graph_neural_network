from utils.classes import *

G = Graph()
G.addNeuron(Neuron(vecteur=cosinus[1]))
G.addNeuron(Neuron(vecteur=sqrt[2]))
G.addNeuron(Neuron(vecteur=sqrt[3]))
G.plotGraph()

G.neurons[0].alterFoyer(cosinus[2])

get_foyer(G, Neuron(vecteur=sqrt[5]))

neuron1 = G.neurons[0].vecteur
neuron2 = G.neurons[1].vecteur
neuron3 = G.neurons[2].vecteur
d12 = distance_neurons(neuron1, neuron2)
d13 = distance_neurons(neuron1, neuron3)
d23 = distance_neurons(neuron2, neuron3)


def func(x):
    return [(x[0] - 0) ** 2 + (x[1] - 0) ** 2 - d13 ** 2,
            (x[0] - d12) ** 2 + (x[1] - 0) ** 2 - d23 ** 2]


root = fsolve(func, [1, 1])
root
from utils.classes import *

G = Graph()
G.addNeuron(Neuron(vecteur=cosinus[1]))
G.addNeuron(Neuron(vecteur=sqrt[2]))
G.addNeuron(Neuron(vecteur=sqrt[3]))

G.neurons[0].alterFoyer(cosinus[2])

get_foyer(G, Neuron(vecteur=sqrt[5]))
from utils.classes import *

G = Graph()
G.addNeuron(Neuron(vecteur=cosinus[1]))
G.addNeuron(Neuron(vecteur=sqrt[2]))

index_foyer, distance_foyer, label_foyer = get_foyer(G, Neuron(vecteur=sqrt[5]))
from utils.classes import *

""" Version article """

# Config 1

# G1 : 0, 3, 4, 6, 7
# G2 : 1, 2, 5, 8, 9

G = Graph()
G.addNeuron(Neuron(vecteur=type_1[1]))  # 0
G.addNeuron(Neuron(vecteur=type_2[2]))  # 1
G.addNeuron(Neuron(vecteur=type_2[3]))  # 2
G.addNeuron(Neuron(vecteur=type_1[2]))  # 3
G.addNeuron(Neuron(vecteur=type_1[4]))  # 4
G.addNeuron(Neuron(vecteur=type_2[5]))  # 5
G.addNeuron(Neuron(vecteur=type_1[3]))  # 6
G.addNeuron(Neuron(vecteur=type_1[5]))  # 7
G.addNeuron(Neuron(vecteur=type_2[1]))  # 8
G.addNeuron(Neuron(vecteur=type_2[4]))  # 9

# Affichage des vecteurs des neurones de cette config
plot_neurons_config_1_article()

print(G.neurons)

# Config 2

# G1 : 0, 2, 4, 7
# G2 : 1, 3, 9, 12
# G3 : 5, 8, 10, 11
# G4 : 6, 13, 14, 15


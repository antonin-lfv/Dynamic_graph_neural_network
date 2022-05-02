from utils.classes import *

G = Graph()
G.addNeuron(Neuron(vecteur=cosinus[1]))
G.addNeuron(Neuron(vecteur=sqrt[2]))
G.addNeuron(Neuron(vecteur=sqrt[3]))
G.addNeuron(Neuron(vecteur=cosinus[2]))
G.addNeuron(Neuron(vecteur=cosinus[4]))
G.addNeuron(Neuron(vecteur=sqrt[5]))
G.addNeuron(Neuron(vecteur=cosinus[3]))
G.addNeuron(Neuron(vecteur=cosinus[5]))
G.addNeuron(Neuron(vecteur=sqrt[1]))
G.addNeuron(Neuron(vecteur=sqrt[4]))
G.plotGraph()



x = [0, 3, -0.4999999999999999, 4.09403388828438]
y = [0, 0, 1.9364916731037085, 2.6847626767436994]
r = [7.6643718897902, 8.8850454548993, 5.6644312314894, 7.6212548653834]
print(solve_inter_circles(x, y, r))

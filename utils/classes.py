from utils.functions import *

class Neuron:
    def __init__(self, vecteur: List, index: int = None, label: str = None, liaison: dict = None):
        """
        :param index: donné par le compteur du graphe
        :param vecteur: les données de taille ConstGraph.INPUT_SIZE
        :param liaison: dictionnaire de liaison avec comme index l'index de l'arrivée et comme valeur le poids synaptique
        """
        if liaison is None:
            liaison = {}
        self.index = index
        self.vecteur = vecteur
        self.liaisons = liaison
        self.label = label

    def __repr__(self):
        return f'Neuron(index={self.index}, vecteur={self.vecteur}, liaison={self.liaisons})'

    def alterFoyer(self):
        """Alteration du neurone dans le cas ou il est le foyer :  Δz = bv*(z-u)"""
        ...

    def alterVoisins(self):
        """Alteration des voisins dans le cas ou il est le foyer : Δxj = bc*cjk(xk-xj)"""
        ...

    def alterLiaisons(self):
        """Alteration des liaisons dans le cas ou il est le foyer : cjk = bl*(||xj-xk||)"""
        ...


class Graph:
    def __init__(self, neurons: dict = None, compt_neurons: int = 0):
        """
        :param neurons: liste des neurones du graphe
        :param compt_neurons: nombre de neurones que le graphe aura vu
        (sert à identifier les neurones à la création)
        """
        if neurons is None:
            neurons = {}
        self.neurons = neurons
        self.compt_neurons = compt_neurons

    def __repr__(self):
        return f'G = Graph(neurons={self.neurons}, compt_neurons={self.compt_neurons})'

    def graphInfo(self):
        print(f"Nombre de neurones vus : {self.compt_neurons}")
        print(f"Nombre de neurones présents dans le réseau : {len(self.neurons)}")

    def plotGraph(self):
        """Plot le Graph actuel avec plotly"""
        fig = go.Figure()

        fig.update_layout(
            showlegend=False,
            xaxis=ConstPlotly.xaxis,
            yaxis=ConstPlotly.yaxis,
            paper_bgcolor=ConstPlotly.transparent_color,
            plot_bgcolor=ConstPlotly.transparent_color,
        )
        plot(fig)

    def addNeuron(self, neuron: Neuron):
        """ Connecte le neurone au réseau
        :param neuron: le neurone à présenter au réseau

        L'altération du foyer, ainsi que des neurones voisins et des
        connexions est faite par des méthodes propres aux neurones.
        Ici on ajoute simplement le neurone et ses connexions.
        """
        # ===== Si il n'y a qu'un seul neurone dans le réseau
        if len(self.neurons) == 1:
            # set index
            neuron.index = self.compt_neurons
            # set label
            neuron.label = str(neuron.index)
            # connexion
            neuron.liaisons = {list(self.neurons.keys())[0]: distance_neurons(list(self.neurons.values())[0].vecteur, neuron.vecteur)}
            # On l'ajoute au réseau
            self.neurons[neuron.index] = neuron
            # on augmente le compteur du graphe
            self.compt_neurons += 1

        # ===== Si il y a au moins 2 neurones dans le réseau
        else:
            # set index
            neuron.index = self.compt_neurons
            # si la distance du foyer est supérieur au seuil, on lui attribut un nouveau label (son index) sinon on lui associe le label du foyer
            index_foyer, distance_foyer, label_foyer = get_foyer(self, neuron)
            if distance_foyer > ConstThreshold.seuilNouveauLabel:
                neuron.label = str(neuron.index)
            else:
                neuron.label = label_foyer
            # On l'ajoute au réseau
            self.neurons[neuron.index] = neuron
            # on augmente le compteur du graphe
            self.compt_neurons += 1
            # connexions

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
        return f'Neuron(index={self.index}, vecteur={"Non affiché"}, liaison={self.liaisons}, label={self.label})'

    def alterFoyer(self):
        """Alteration du neurone dans le cas ou il est le foyer :  Δz = bv*(z-u)"""
        ...

    def alterVoisins(self):
        """Alteration des voisins dans le cas ou il est le foyer : Δxj = bc*cjk(xk-xj)"""
        ...

    def alterLiaisons(self):
        """Alteration des liaisons dans le cas ou il est le foyer : cjk = bl*(||xj-xk||)
        C'est à ce moment là qu'on peut décider de couper des liaisons si le poids est supérieur à ar"""
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
        # ===== Si il n'y a pas de neurone
        # Aucune liaison n'est créée
        if len(self.neurons) == 0:
            # set index
            neuron.index = self.compt_neurons
            # set label
            neuron.label = str(neuron.index)
            # On l'ajoute au réseau
            self.neurons[neuron.index] = neuron
            # on augmente le compteur du graphe
            self.compt_neurons += 1

        else:
            # set index
            neuron.index = self.compt_neurons
            # set label
            # si la distance du foyer est supérieur au seuil an, on lui attribut un nouveau label (son index) sinon on lui associe le label du foyer
            foyer = get_foyer(self, neuron)
            if distance_neurons(foyer.vecteur, neuron.vecteur) > ConstThreshold.an:
                neuron.label = str(neuron.index)
            else:
                neuron.label = foyer.label
            # set connexions
            if len(self.neurons) == 1:
                # ===== Il y a un seul neurone dans le réseau -> création d'une seule connexion
                # on les connecte forcément pour éviter un arret instantané à cause du seuil de suppression des liaisons
                neuron.liaisons[foyer.index] = foyer.liaisons[neuron.index] = distance_neurons(foyer.vecteur, neuron.vecteur)

            else:
                # ===== Il y a au moins 2 neurones dans le réseau
                for n in self.neurons.values():
                    print(n)
                    if distance_neurons(foyer.vecteur, neuron.vecteur) < ConstThreshold.an:
                        # Si la distance du foyer est plus petite que an on connecte à tous les neurones de distance < an
                        ...
                    else:
                        # Si la distance du foyer est supérieur à an on connecte le neurone seulement au foyer
                        # neuron.liaisons[foyer.index] = foyer.liaisons[neuron.index] = distance_neurons(foyer.vecteur, neuron.vecteur)
                        ...

            # On l'ajoute au réseau
            self.neurons[neuron.index] = neuron
            # on augmente le compteur du graphe
            self.compt_neurons += 1
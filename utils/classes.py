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
        return f'Neuron(index={self.index}, vecteur="Pas d\'affichage", liaison={self.liaisons}, label={self.label})'

    def alterFoyer(self, u: List[float]):
        """Alteration du neurone dans le cas ou il est le foyer :  Δz = bv*(z-u)"""
        Deltaz = [ConstThreshold.bv * (a + b) for a, b in zip(self.vecteur, u)]
        self.vecteur = [a + b for a, b in zip(self.vecteur, Deltaz)]

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
        # Création des points
        neuron_points_x = []
        neuron_points_y = []
        neuron_points_label = []
        for index, n in self.neurons.items():
            neuron_points_x.append(index)
            neuron_points_y.append(0)
            neuron_points_label.append(f'Label = {n.label}')
        fig.add_scatter(x=neuron_points_x, y=neuron_points_y, text=neuron_points_label, mode='markers+text',
                        hovertemplate="<b>%{text}</b><extra></extra>", textposition="bottom center",
                        textfont=dict(
                            size=10,
                        ),
                        marker=dict(
                            color='black'
                        ))

        # Création des liaisons


        fig.update_layout(
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
            # get foyer
            foyer = get_foyer(self, neuron)
            # set connexions
            if len(self.neurons) == 1:
                # ===== Il y a un seul neurone dans le réseau -> création d'une seule connexion
                # Le label est attribué avec le seuil an
                if distance_neurons(foyer.vecteur, neuron.vecteur) > ConstThreshold.an:
                    neuron.label = str(neuron.index)
                else:
                    neuron.label = foyer.label
                # on les connecte forcément pour éviter un arret instantané à cause du seuil de suppression des liaisons
                neuron.liaisons[foyer.index] = foyer.liaisons[neuron.index] = distance_neurons(foyer.vecteur,
                                                                                               neuron.vecteur)

            else:
                # ===== Il y a au moins 2 neurones dans le réseau
                if distance_neurons(foyer.vecteur, neuron.vecteur) < ConstThreshold.an:
                    # set label
                    neuron.label = foyer.label
                    # Si la distance du foyer est plus petite que an on connecte à tous les neurones de distance < an
                    for n in self.neurons.values():
                        if (d := distance_neurons(n.vecteur, neuron.vecteur)) < ConstThreshold.an:
                            neuron.liaisons[n.index] = n.liaisons[neuron.index] = d
                else:
                    # set label
                    neuron.label = str(neuron.index)
                    # Si la distance du foyer est supérieur à an on connecte le neurone seulement au foyer
                    neuron.liaisons[foyer.index] = foyer.liaisons[neuron.index] = distance_neurons(foyer.vecteur,
                                                                                                   neuron.vecteur)

            # On l'ajoute au réseau
            self.neurons[neuron.index] = neuron
            # on augmente le compteur du graphe
            self.compt_neurons += 1

import numpy as np

from utils.functions import *


class Neuron:
    def __init__(self, vecteur: List, config, index: int = None, label: str = None, liaisons: dict = None):
        """
        :param index: donné par le compteur du graphe
        :param config: config du graphe
        :param vecteur: les données de taille ConstGraph.INPUT_SIZE
        :param liaisons: dictionary de liaison avec comme index l'index de l'arrivée et comme valeur le poids synaptique
        """
        if liaisons is None:
            liaisons = {}
        self.index = index
        self.vecteur = vecteur
        self.config = config
        self.liaisons = liaisons
        self.label = label

    def __repr__(self):
        return f'Neuron(index={self.index}, vecteur="", liaisons={self.liaisons}, label={self.label})'

    def alterFoyer(self, u: List[float]):
        """Alteration du neurone dans le cas ou il est le foyer :  Δz = bv*(z-u)"""
        Deltaz = [self.config["bv"] * (a - b) for a, b in zip(self.vecteur, u)]
        self.vecteur = [a + b for a, b in zip(self.vecteur, Deltaz)]

    def alterVoisins(self, graph):
        """Alteration des voisins dans le cas ou il est le foyer : Δxj = bc*cjk(xk-xj)"""
        for k, val in self.liaisons.items():
            # k prend les valeurs des index des neurones voisins, donc de similarité < an
            graph.neurons[k].vecteur = [i + j for i, j in zip(graph.neurons[k].vecteur,
                                                              [self.config["bc"] * val * (a - b) for a, b in
                                                               zip(self.vecteur, graph.neurons[k].vecteur)])]

    def alterLiaisons(self, graph):
        """Alteration des liaisons dans le cas ou il est le foyer : cjk = bl*(||xj-xk||)"""
        for k, val in self.liaisons.items():
            if (tailleLiaison := self.liaisons[k] * self.config["bl"]) < self.config["ar"]:
                self.liaisons[k] = graph.neurons[k].liaisons[self.index] = tailleLiaison


class Graph:
    def __init__(self, config, fct_distance: Callable = None, neurons: dict = None, compt_neurons: int = 0,
                 suppr_neuron=False):
        """
        :param config: seuils du modèle
        :param neurons: liste des neurones du graphe
        :param compt_neurons: nombre de neurones que le graphe aura vu
        (sert à identifier les neurones à la création)
        """
        if neurons is None:
            neurons = {}
        self.neurons = neurons
        self.config = config
        self.compt_neurons = compt_neurons
        if fct_distance is None:
            # default - euclidean distance
            self.fct_distance = distance_neurons
        else:
            self.fct_distance = fct_distance
        self.suppr_neuron = suppr_neuron

    def get_neuron_index(self):
        """Retourne les index des neurones du graphe"""
        return [n.index for n in self.neurons.values()]

    def get_vector_as_dict(self):
        """retourne un dictionnaire avec les vecteurs des neurones"""
        r, index = {}, 0
        for neuron in self.neurons.values():
            r[index] = neuron.vecteur
            index += 1
        return r

    def get_size_of_links(self, plotly_fig=True):
        """Retourne la taille des liaisons du graphe"""
        r = []
        for neuron in self.neurons.values():
            for val in neuron.liaisons.values():
                r.append(val)
        if plotly_fig:
            fig = go.Figure(data=[go.Box(y=r,
                                         boxpoints='all',  # can also be outliers, or suspectedoutliers, or False
                                         jitter=0.3,  # add some jitter for a better separation between points
                                         pointpos=-1.8  # relative position of points wrt box
                                         )])
            plot(fig)
        return r

    def __repr__(self):
        return f'G = Graph(neurons={self.neurons}, compt_neurons={self.compt_neurons})'

    def graphInfo(self):
        print(colored("\n===== Info Graphe", "red"))
        print(f"Nombre de neurones vus : {self.compt_neurons}")
        print(f"Nombre de neurones présents dans le réseau : {len(self.neurons)}")

    def addNeuron(self, neuron: Neuron):
        """ Connecte le neurone au réseau
        :param neuron: le neurone à présenter au réseau

        L'altération du foyer, ainsi que des neurones voisins et des
        connexions est faite par des méthodes propres aux neurones.
        """
        # ===== Si il n'y a pas de neurone
        # Aucune liaison n'est créée
        if len(self.neurons) == 0:
            # set index
            if neuron.index is None:
                neuron.index = self.compt_neurons
            # set label
            neuron.label = str(neuron.index)
            # On l'ajoute au réseau
            self.neurons[neuron.index] = neuron
            # on augmente le compteur du graphe
            self.compt_neurons += 1

        else:
            # set index
            if neuron.index is None:
                neuron.index = self.compt_neurons
            # get foyer
            foyer = get_foyer(self, neuron)
            # set connexions
            if len(self.neurons) == 1:
                # ===== Il y a un seul neurone dans le réseau -> création d'une seule connexion
                # Le label est attribué avec le seuil an
                if self.fct_distance(foyer.vecteur, neuron.vecteur) > self.config["an"]:
                    neuron.label = str(neuron.index)
                else:
                    neuron.label = foyer.label
                # on les connecte forcément pour éviter un arret instantané à cause du seuil de suppression des liaisons
                neuron.liaisons[foyer.index] = foyer.liaisons[neuron.index] = self.fct_distance(foyer.vecteur,
                                                                                                neuron.vecteur)

            else:
                # ===== Il y a au moins 2 neurones dans le réseau
                if self.fct_distance(foyer.vecteur, neuron.vecteur) < self.config["an"]:
                    # set label
                    neuron.label = foyer.label
                    # Si la distance du foyer est plus petite que an on connecte à tous les neurones de distance < an
                    for n in self.neurons.values():
                        if (d := self.fct_distance(n.vecteur, neuron.vecteur)) < self.config["an"]:
                            neuron.liaisons[n.index] = n.liaisons[neuron.index] = d
                else:
                    # set label
                    neuron.label = str(neuron.index)
                    # Si la distance du foyer est supérieur à an on connecte le neurone seulement au foyer
                    neuron.liaisons[foyer.index] = foyer.liaisons[neuron.index] = self.fct_distance(foyer.vecteur,
                                                                                                    neuron.vecteur)
            # On l'ajoute au réseau
            self.neurons[neuron.index] = neuron
            # on augmente le compteur du graphe
            self.compt_neurons += 1
            # on altère le foyer seulement si le neurone est très proche du foyer, cad d<an
            if self.fct_distance(foyer.vecteur, neuron.vecteur) < self.config["an"]:
                foyer.alterFoyer(neuron.vecteur)
                foyer.alterVoisins(self)
                foyer.alterLiaisons(self)

        # suppression des liaisons trop grandes
        for n in self.neurons.values():
            suppr_ = []
            for k, i in n.liaisons.items():
                if i > self.config["ar"]:
                    suppr_.append(k)
            for ind_suppr in suppr_:
                del n.liaisons[ind_suppr]

        # Suppression du neurone si suppr_neuron à True (on commence dès qu'il y a 2 neurones)
        suppr_list = []
        for n in self.neurons.values():
            if len(n.liaisons) == 0 and len(self.neurons) > 1:
                if not self.suppr_neuron:
                    # Cas 1 (défaut) : si un neurone n'a plus de liaison, il est associé à un label à part
                    n.label = str(n.index)
                else:
                    # Cas 2 : si un neurone n'a plus de liaison, il est supprimé
                    suppr_list.append(n.index)
        for supp in suppr_list:
            del self.neurons[supp]

    def fit(self, X: dict, print_progress=True, use_existing_index=False):
        """ Ajout des neurones - Un seul ajout de tous les neurones
        :param X : Ensemble de signaux sous forme de dictionnaire
        :param use_existing_index: Utilise l'index du neurone utilisé dans le dictionnaire passé en paramètre
        (mettre à True pour les dico mélangés)
        """
        compt = 0
        if print_progress:
            print(colored("\n===== Début de l'apprentissage", "red"))
            print("Ajout des", colored(f"{len(X)}", "green"), "neurones")
        for index, x in X.items():
            if use_existing_index:
                self.addNeuron(Neuron(vecteur=x, config=self.config, index=index))
            else:
                self.addNeuron(Neuron(vecteur=x, config=self.config))
            if print_progress:
                print(colored(f"[{compt}/{len(X) - 1}]", "green") + " Neurone ajouté !")
            compt += 1

    def print_cluster(self, display=False):
        """retourne la composition des cluster (console)
        @:param display: True si affichage, sinon simple return du dict
        """
        clusters = {}
        for n in self.neurons.values():
            if n.label in clusters.keys():
                clusters[n.label].append(n.index)
            else:
                clusters[n.label] = [n.index]
        if display:
            print(colored("\n===== Résultat de la classification", "red"))
            for label, neurons in clusters.items():
                print(f"Label {label} : ", *neurons)
            return 1
        return clusters


# TODO : plutot afficher le graphe en rassemblant chaque label dans un seul neurone
# TODO : on ne les connecte qu'avec les autres labels

# TODO : ajouter une phase d'optimisation
# TODO : optimiser soit le nombre de classes finales
# TODO : soit une autre métrique


"""idée : méthode de Graph pour afficher le graphe avec plotly"""


# TODO : trier les neurones par nombre de liaisons, calculer l'intersection uniquement avec les liaisons existantes
# TODO : commencer par le neurone avec le plus de liaisons, puis le suivant ...
# TODO : définir l'ordre de passage des points


def plotGraph(self):
    """Plot le Graph actuel avec plotly"""
    fig = go.Figure()
    # Création des points
    neuron_points_x = []
    neuron_points_y = []
    neuron_points_info = []
    neuron_index = []
    # Création des liaisons inter-neurones
    liaison_x = []
    liaison_y = []
    # index des neurones existants
    index_n = self.get_neuron_index()
    for index, n in self.neurons.items():
        if index == index_n[0]:
            # 1er point à placer
            neuron_points_x.append(0)
            neuron_points_y.append(0)
            neuron_points_info.append(f'Classe = {n.label}<br>Neurone {n.index}')
            neuron_index.append(n.index)
        elif index == index_n[1]:
            # 2e point à placer par rapport au premier
            neuron_points_x.append(distance_neurons(n.vecteur, self.neurons[index_n[0]].vecteur))
            neuron_points_y.append(0)
            liaison_x.extend([neuron_points_x[-1], neuron_points_x[-2], None])
            liaison_y.extend([neuron_points_y[-1], neuron_points_y[-2], None])
            neuron_points_info.append(f'Classe = {n.label}<br>Neurone {n.index}')
            neuron_index.append(n.index)
        else:
            # jème point à placer par rapport aux j-1 premiers,
            # intersection de j-1 cercles
            x, y = solve_inter_circles(neuron_points_x, neuron_points_y,
                                       [distance_neurons(self.neurons[i].vecteur, n.vecteur) for i in index_n])
            neuron_points_x.append(x)
            neuron_points_y.append(y)
            neuron_index.append(n.index)
            for index_liaison, liaisons in n.liaisons.items():
                if index_liaison in neuron_index:
                    liaison_x.extend(
                        [neuron_points_x[-1], neuron_points_x[neuron_index.index(index_liaison)], None])
                    liaison_y.extend(
                        [neuron_points_y[-1], neuron_points_y[neuron_index.index(index_liaison)], None])
            neuron_points_info.append(f'Classe = {n.label}<br>Neurone {n.index}')

    fig.add_scatter(x=neuron_points_x, y=neuron_points_y, text=neuron_points_info, mode='markers+text',
                    hovertemplate="<b>%{text}</b><extra></extra>", textposition="bottom center",
                    textfont=dict(
                        size=10,
                    ),
                    marker=dict(
                        color='black'
                    ))

    fig.add_scatter(x=liaison_x, y=liaison_y, mode="lines", opacity=.2, line=dict(color='blue'))

    fig.update_layout(
        xaxis=ConstPlotly.xaxis,
        yaxis=ConstPlotly.yaxis,
        paper_bgcolor=ConstPlotly.transparent_color,
        plot_bgcolor=ConstPlotly.transparent_color,
        showlegend=False
    )
    plot(fig, filename='plot.html')
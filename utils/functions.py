from utils.const import *

"""Fonctions pour le graphe"""


def distance_neurons(x: list, y: list) -> float:
    """ Distance euclidienne entre 2 vecteurs de neurones
    :param x: vecteur du premier neurone de taille l
    :param y: vecteur du deuxieme neurone de taille l
    """
    return round(fastdist.euclidean(np.array(x), np.array(y)), 3)


def get_foyer(graph, neuron):
    """Retourne l'index, la distance et le label du foyer d'un neurone d'entrée
    :param graph:
    :param neuron: le neurone d'entrée
    """
    if len(graph.neurons) != 0:
        distance_foyer, foyer = np.inf, graph.neurons[list(graph.neurons.keys())[0]]
        for n in graph.neurons.keys():
            if (d := distance_neurons(neuron.vecteur, graph.neurons[n].vecteur)) < distance_foyer:
                foyer = graph.neurons[n]
                distance_foyer = d
        return foyer
    else:
        raise ValueError("Le graphique ne contient aucun neurone")


def solve_inter_circles(centres_x, centres_y, rayons):
    """Get one intersection of n circles
    :param centres_x: liste des abscisses des n centres des cercles
    :param centres_y: liste des ordonnées des n centres des cercles
    :param rayons: liste des rayons des n cercles
    """

    def func(x):
        return [(x[0] - cx) ** 2 + (x[1] - cy) ** 2 - d ** 2 for cx, cy, d in zip(centres_x, centres_y, rayons)]

    root = fsolve(func, np.array([1] * len(centres_x)), maxfev=500)
    return root[0], root[1]


"""Fonctions pour les signaux"""


def dict_of_signal(abscisse, nb_neurons):
    """Retourne un dictionnaire de signaux
    @:param abscisse : la liste des abscisses
    @:param nb_neurons : le nombre de neurones"""
    # fonctions
    s = np.sin
    c = np.cos
    # constantes
    pi = np.pi

    def random_signal():
        signal = (-1) ** random.randint(1, 2) * random.uniform(0, 1) * s(abscisse)
        common = random.randint(1, 3)
        for _ in range(11):
            signal += (-1) ** random.randint(1, 2) * random.uniform(0, 3) * c(
                pi * random.uniform(1, 3) * common * abscisse) + (
                          -1) ** random.randint(1, 2) * random.uniform(0, 3) * s(
                pi * random.uniform(1, 3) * common * abscisse)
        return signal

    fct = {}
    for i in range(nb_neurons):
        fct[i] = random_signal()
    return fct


def plot_dict_signal(abs, dict_y, signaux, nb_neurons):
    """Affiche un dictionnaire de signaux
    @:param abs: l'abscisse
    @:param dict_y: le dict de signaux
    """
    size = math.ceil(np.sqrt(len(signaux)))
    fig = make_subplots(rows=size, cols=size, subplot_titles=[f"Neurone {i}" for i in range(len(signaux))])
    index_signal = 0
    for row in range(1, size + 1):
        for col in range(1, size + 1):
            if index_signal < nb_neurons:
                fig.add_scatter(row=row, col=col, x=abs, y=dict_y[index_signal], name=index_signal)
                index_signal += 1
    fig.update_layout(
        paper_bgcolor=ConstPlotly.transparent_color,
        showlegend=False
    )
    plot(fig)


def print_cluster(G, display):
    """retourne la composition des cluster
    @:param G: le graphe
    @:param display: True si affichage, sinon simple return du dict
    """
    clusters = {}
    for n in G.neurons.values():
        if n.label in clusters.keys():
            clusters[n.label].append(n.index)
        else:
            clusters[n.label] = [n.index]
    if display:
        for label, neurons in clusters.items():
            print(f"Label {label} : ", *neurons)
    return clusters


def plot_signaux_par_cluster(G, abs, dict_y):
    """
    :param G: le graphe
    :param abs: abs_normal si plot les signaux brutes, sinon abs_fft pour plot les signaux après FFT
    :param dict_y: le dictionnaire des signaux brutes ou FFT
    :return:
    """
    clusters = print_cluster(G, display=False)
    fig = make_subplots(rows=max([len(i) for i in clusters.values()]), cols=len(clusters),
                        column_titles=[f"Label {i}" for i in clusters.keys()])
    for label in clusters.keys():
        row_index = 1
        for neuron_index in clusters[label]:
            fig.add_scatter(row=row_index, col=list(clusters.keys()).index(label) + 1, x=abs, y=dict_y[neuron_index],
                            text=f"Index : {neuron_index}", hoverinfo="text")
            row_index += 1
    fig.update_layout(
        paper_bgcolor=ConstPlotly.transparent_color,
        showlegend=False
    )
    plot(fig)


def dict_of_wv(signaux):
    """Retourne un dictionnaire de wavelet correspondant aux signaux"""
    wv_dict = {}
    for index_wv, s in enumerate(signaux.values()):
        wv_dict[index_wv] = 2.0 / ConstGraph_article.INPUT_SIZE_CONFIG_3 * np.abs(
            fft(s)[0:ConstGraph_article.INPUT_SIZE_CONFIG_3 // 2])
    return wv_dict


def dict_of_fft(signaux):
    """Retourne un dictionnaire de fft correspondant aux signaux"""
    fft_dict = {}
    for index_fft, s in enumerate(signaux.values()):
        fft_dict[index_fft] = 2.0 / ConstGraph_article.INPUT_SIZE_CONFIG_3 * np.abs(
            fft(s)[0:ConstGraph_article.INPUT_SIZE_CONFIG_3 // 2])
    return fft_dict


"""Lire les ressources """

import pickle

from utils.const import *

"""Fonctions pour le graphe"""


def distance_neurons(x: list, y: list) -> float:
    """ Distance euclidienne entre 2 vecteurs de neurones
    :param x: vecteur du premier neurone de taille l
    :param y: vecteur du deuxieme neurone de taille l
    """
    return round(fastdist.euclidean(np.array(x), np.array(y)), 3)


def distance_neurons_DTW(x: list, y: list) -> float:
    """ Distance euclidienne entre 2 vecteurs de neurones
    :param x: vecteur du premier neurone de taille m
    :param y: vecteur du deuxième neurone de taille n
    """
    distance, _ = fastdtw(x, y, dist=euclidean)
    return round(distance, 3)

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


def plot_dict_signal(dict_y, nb_neurons, absc=None):
    """Affiche un dictionnaire de signaux
    @:param abs: l'abscisse
    @:param dict_y: le dict de signaux
    """
    size = math.ceil(np.sqrt(len(dict_y)))
    fig = make_subplots(rows=size, cols=size, subplot_titles=[f"Neurone {i}" for i in range(len(dict_y))])
    index_signal = 0
    for row in range(1, size + 1):
        for col in range(1, size + 1):
            if index_signal < nb_neurons:
                if absc is None:
                    fig.add_scatter(row=row, col=col, x=absc, y=dict_y[index_signal], name=index_signal)
                else:
                    fig.add_scatter(row=row, col=col, y=dict_y[index_signal], name=index_signal)
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


def plot_signaux_par_cluster(G, dict_y, absc=None):
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
            if absc is None:
                fig.add_scatter(row=row_index, col=list(clusters.keys()).index(label) + 1, y=dict_y[neuron_index],
                                text=f"Index : {neuron_index}", hoverinfo="text")
            else:
                fig.add_scatter(row=row_index, col=list(clusters.keys()).index(label) + 1, x=absc,
                                y=dict_y[neuron_index],
                                text=f"Index : {neuron_index}", hoverinfo="text")
            row_index += 1
    fig.update_layout(
        paper_bgcolor=ConstPlotly.transparent_color,
        showlegend=False
    )
    plot(fig)


def dict_of_fft(signaux, taille_signaux=None):
    """
    Retourne un dictionnaire de fft correspondant aux signaux
    """
    fft_dict = {}
    for index_fft, s in enumerate(signaux.values()):
        if taille_signaux:
            fft_dict[index_fft] = 2.0 / taille_signaux * np.abs(
                fft(s)[0:taille_signaux // 2])
        else:
            fft_dict[index_fft] = 2.0 / len(s) * np.abs(
                fft(s)[0:len(s) // 2])
    return fft_dict


def plot_rapide(y):
    fig = go.Figure()
    fig.add_scatter(y=y)
    plot(fig)


"""Lire les ressources - chants d'oiseaux"""


# data : https://figshare.com/articles/media/BirdsongRecognition/3470165?file=5463221


def read(path):
    y, _ = librosa.load(path)
    return y


def create_dict_of_birds():
    """
    Création des partitions des chants d'oiseaux et remplissage des dictionnaires
    """
    birds = {}
    with open('data/Birdsong/birdsongs.pkl', 'wb') as f:
        birds[0] = read("data/Birdsong/Bird0/Wave/3.wav")[48900:50700]
        birds[1] = read("data/Birdsong/Bird0/Wave/3.wav")[50900:52750]
        birds[2] = read("data/Birdsong/Bird0/Wave/3.wav")[52976:54800]
        birds[3] = read("data/Birdsong/Bird0/Wave/3.wav")[55000:57000]
        birds[4] = read("data/Birdsong/Bird0/Wave/3.wav")[57280:59200]
        birds[5] = read("data/Birdsong/Bird1/Wave/5.wav")[93381:95500]
        birds[6] = read("data/Birdsong/Bird1/Wave/5.wav")[96000:98300]
        birds[7] = read("data/Birdsong/Bird1/Wave/5.wav")[96000:98300]
        birds[8] = read("data/Birdsong/Bird1/Wave/5.wav")[98800:101000]
        birds[9] = read("data/Birdsong/Bird1/Wave/5.wav")[101500:103600]
        birds[10] = read("data/Birdsong/Bird2/Wave/0.wav")[25700:94727]
        birds[11] = read("data/Birdsong/Bird2/Wave/0.wav")[99180:160300]
        birds[12] = read("data/Birdsong/Bird2/Wave/0.wav")[164300:206900]
        birds[13] = read("data/Birdsong/Bird2/Wave/0.wav")[210500:256255]
        birds[14] = read("data/Birdsong/Bird2/Wave/0.wav")[257000:309000]
        pickle.dump(birds, f)

    with open('data/Birdsong/corr.pkl', 'wb') as f:
        corr = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 1,
            6: 1,
            7: 1,
            8: 1,
            9: 1,
            10: 2,
            11: 2,
            12: 2,
            13: 2,
            14: 2
        }
        pickle.dump(corr, f)


def dict_of_birds():
    """
    Retourne un dictionnaire de 15 chants d'oiseaux et un dictionnaire de correspondance avec la classe d'oiseau
    Ici, les signaux 0 à 4 seront ceux de l'oiseau 0, 5 à 9 ceux de l'oiseau 1 et 10 à 14 de l'oiseau 2
    """
    birds_dict = pickle.load(open('data/Birdsong/birdsongs.pkl', 'rb'))
    corr_dict = pickle.load(open('data/Birdsong/corr.pkl', 'rb'))

    return birds_dict, corr_dict

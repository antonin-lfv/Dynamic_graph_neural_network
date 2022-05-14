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


"""Lire les ressources - chants des oiseaux"""


def read(file_path):
    x, sr = a2n.audio_from_file(file_path)
    return x


def dict_of_birds():
    """
    Retourne un dict de signaux et un dict de correspondance à la classe réelle
    Ici, on aura 5 syllabes de chants d'oiseaux de 3 espèces différentes
    """
    # Path
    moineau_friquet = read("data/birdsong/Bird song 1/Moineau friquet/EurasianTreeSparrow14April2009Dwingelderveld.mp3")
    pinsons_du_nord = read("data/birdsong/Bird song 1/Pinson du Nord/FrinMont song male 22609 0347.mp3")
    pinsons_des_arbres = read("data/birdsong/Bird song 1/Pinson des arbres/XC113679-Fringilla_colebs_Estonia_Jarek_Matusiak_20100430-004.mp3")
    # création des dictionnaires
    signaux = {0: moineau_friquet[521000:524500], 1: moineau_friquet[119000:122000], 2: moineau_friquet[942000:945000],
               3: moineau_friquet[19500:23000], 4: moineau_friquet[323000:327000]}
    ...
    corr_classes = {
        0: "moineau_friquet",
        1: "moineau_friquet",
        2: "moineau_friquet",
        3: "moineau_friquet",
        4: "moineau_friquet",
    }
    return signaux, corr_classes


"""
Moineau friquet:
> EurasianTreeSparrow14April2009Dwingelderveld.mp3
521k - 524.5k
119k - 122k
942k - 945k
19.5k - 23k
323k - 327k

pinsons du nord:
> FrinMont song male 22609 0347.mp3
53k - 84k
291k - 322k
491k - 521k
711k - 740k
929k - 958k

pinsons des arbres:
> XC113679-Fringilla_colebs_Estonia_Jarek_Matusiak_20100430-004.mp3
63k - 159k
357k - 465k
710k - 800k
1130k - 1245k
1371k - 1494k
"""

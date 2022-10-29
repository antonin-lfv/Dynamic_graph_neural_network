import random

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
    """Génère un dictionnaire de signaux
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
    """Affiche un dictionnaire de signaux (plotly)
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
    """retourne la composition des cluster (console)
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
        print(colored("\n===== Résultat de la classification", "red"))
        for label, neurons in clusters.items():
            print(f"Label {label} : ", *neurons)
    return clusters


def plot_signaux_par_cluster(G, dict_y, absc=None, sign_min_per_cluster=1):
    """
    :param G: le graphe après fit()
    :param absc: abs_normal si plot les signaux brutes, sinon abs_fft pour plot les signaux après FFT, etc
    :param dict_y: le dictionnaire des signaux
    :param sign_min_per_cluster: minimum de signaux par cluster pour être affiché
    :return:
    """
    clusters = print_cluster(G, display=False)
    # Garder les clusters avec au moins sign_min_per_cluster signaux
    clusters = dict(filter(lambda elem: len(elem[1]) >= sign_min_per_cluster, clusters.items()))
    fig = make_subplots(rows=max([len(i) for i in clusters.values()]),
                        cols=len(clusters),
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
        showlegend=False,
        title=f"Ne sont affichés que les clusters d'au moins {sign_min_per_cluster} signaux"
    )
    plot(fig)


def dict_of_fft(signaux: dict, taille_signaux=None):
    """
    :param signaux: signaux brutes
    Retourne un dictionnaire de fft correspondant aux signaux
    """
    fft_dict = {}
    for index_fft, s in signaux.items():
        if taille_signaux:
            fft_dict[index_fft] = 2.0 / taille_signaux * np.abs(
                fft(s)[0:taille_signaux // 2])
        else:
            fft_dict[index_fft] = 2.0 / len(s) * np.abs(
                fft(s)[0:len(s) // 2])
    return fft_dict


def dict_of_wavelet(signaux: dict, scale: int = 3):
    """
    :param scale: scale de l'ondelette
    :param signaux: signaux brutes
    Retourne un dictionnaire d'ondelettes correspondant aux signaux
    """
    wavelet_dict = {}
    for index_wv, s in signaux.items():
        cA, cD = pywt.cwt(s, wavelet='morl', scales=np.arange(1, 129))
        wavelet_dict[index_wv] = cA[:, scale]
    return wavelet_dict


def normalize_dict_values(d: dict):
    """
    Normalise les vecteurs 1D d'un dictionnaire
    :param d: le dictionnaire de données
    """
    for ind, val in d.items():
        d[ind] = normalize(val.reshape(1, -1)).ravel()
    return d


def plot_rapide(y, many=False):
    """
    :param y: data
    :param many: if True, y must contains multiple array to plot
    """
    fig = make_subplots(rows=len(y) if many == True else 1, cols=1)
    if not many:
        fig.add_scatter(y=y)
    else:
        index = 1
        for data in y:
            fig.add_scatter(y=data, row=index, col=1)
            index += 1
    plot(fig)


def plot_rapide_dash(y, many=False):
    app = Dash(__name__)
    fig = go.Figure()
    if not many:
        # Un seul
        fig.add_scatter(y=y)
        f = dcc.Graph(figure=fig)
    else:
        # pleins
        for data in y:
            fig.add_scatter(y=data)
        f = [dcc.Graph(figure=fig)]

    app.layout = html.Div(children=[
        html.H1(children='Graphiques'),

        html.Div(children='''
            Signaux
        '''),
        f
    ])
    app.run_server(debug=True)


def shuffle_dict(dico: dict, seed=1):
    """
    :param dico: Dictionnaire à mélanger
    :param seed: graine de mélange, pour mélanger de la même façon les valeurs et les clés
    :return: le dictionnaire mélangé
    """
    # shuffling values and keys
    values = list(dico.values())
    keys = list(dico.keys())
    random.Random(seed).shuffle(values)
    random.Random(seed).shuffle(keys)
    # reassigning to keys
    res = dict(zip(keys, values))
    return res


def print_config(config: dict):
    print(colored("\n===== Config", "red"))
    for k, i in config.items():
        print(f"{k}: {i}")


"""Python fonctions"""


def get_file_in_folder(folder_path):
    """Retourne la liste des fichiers d'un dossier"""
    return glob.glob(folder_path + "/*")


def numpy_from_matlab(path_matFile):
    """Converti un fichier .mat en numpy array"""
    mat = scipy.io.loadmat(path_matFile)['val']
    return mat.reshape((len(mat[0], )))


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


def create_dict_of_ECG():
    """
    Extraction des pulsations des ECG et remplissage des dictionnaires
    Chaque pulsation sera prise sur une durée de 250
    10 pulsations par classe
    Classes présentes :
    - NSR index de 0 à 9
    - APB index de 10 à 19
    - AFL index de 20 à 29
    """
    ECG = {}
    with open('data/ECG_signals/ECG.pkl', 'wb') as f:
        # NSR
        ECG[0] = numpy_from_matlab('data/ECG_signals/1 NSR/105m (7).mat')[1050:1300]
        ECG[1] = numpy_from_matlab('data/ECG_signals/1 NSR/105m (7).mat')[1300:1550]
        ECG[2] = numpy_from_matlab('data/ECG_signals/1 NSR/105m (7).mat')[1550:1800]
        ECG[3] = numpy_from_matlab('data/ECG_signals/1 NSR/105m (7).mat')[1800:2050]
        ECG[4] = numpy_from_matlab('data/ECG_signals/1 NSR/105m (7).mat')[2050:2300]
        ECG[5] = numpy_from_matlab('data/ECG_signals/1 NSR/105m (7).mat')[2300:2550]
        ECG[6] = numpy_from_matlab('data/ECG_signals/1 NSR/100m (2).mat')[1050:1300]
        ECG[7] = numpy_from_matlab('data/ECG_signals/1 NSR/100m (2).mat')[1300:1550]
        ECG[8] = numpy_from_matlab('data/ECG_signals/1 NSR/100m (2).mat')[1550:1800]
        ECG[9] = numpy_from_matlab('data/ECG_signals/1 NSR/100m (2).mat')[1800:2050]
        # APB
        ECG[10] = numpy_from_matlab('data/ECG_signals/2 APB/100m (2).mat')[1050:1300]
        ECG[11] = numpy_from_matlab('data/ECG_signals/2 APB/100m (2).mat')[1300:1550]
        ECG[12] = numpy_from_matlab('data/ECG_signals/2 APB/100m (2).mat')[1550:1800]
        ECG[13] = numpy_from_matlab('data/ECG_signals/2 APB/100m (2).mat')[1800:2050]
        ECG[14] = numpy_from_matlab('data/ECG_signals/2 APB/100m (2).mat')[2050:2300]
        ECG[15] = numpy_from_matlab('data/ECG_signals/2 APB/100m (2).mat')[2200:2450]
        ECG[16] = numpy_from_matlab('data/ECG_signals/2 APB/100m (3).mat')[1100:1350]
        ECG[17] = numpy_from_matlab('data/ECG_signals/2 APB/103m (0).mat')[950:1200]
        ECG[18] = numpy_from_matlab('data/ECG_signals/2 APB/103m (0).mat')[1250:1500]
        ECG[19] = numpy_from_matlab('data/ECG_signals/2 APB/103m (0).mat')[1600:1850]
        # AFL
        ECG[20] = numpy_from_matlab('data/ECG_signals/3 AFL/202m (0).mat')[1300:1550]
        ECG[21] = numpy_from_matlab('data/ECG_signals/3 AFL/202m (0).mat')[1800:2050]
        ECG[22] = numpy_from_matlab('data/ECG_signals/3 AFL/202m (0).mat')[2000:2250]
        ECG[23] = numpy_from_matlab('data/ECG_signals/3 AFL/202m (0).mat')[2650:2900]
        ECG[24] = numpy_from_matlab('data/ECG_signals/3 AFL/202m (0).mat')[2950:3200]
        ECG[25] = numpy_from_matlab('data/ECG_signals/3 AFL/222m (10).mat')[1330:1580]
        ECG[26] = numpy_from_matlab('data/ECG_signals/3 AFL/222m (10).mat')[2600:2850]
        ECG[27] = numpy_from_matlab('data/ECG_signals/3 AFL/222m (10).mat')[3050:3300]
        ECG[28] = numpy_from_matlab('data/ECG_signals/3 AFL/203m (2).mat')[400:650]
        ECG[29] = numpy_from_matlab('data/ECG_signals/3 AFL/203m (2).mat')[1000:1250]
        pickle.dump(ECG, f)

    with open('data/ECG_signals/corr.pkl', 'wb') as f:
        corr = {}
        classes = ["NSR", "APB", "AFL"]
        for i in range(len(ECG)):
            corr.update({i: classes[i//10]})
        pickle.dump(corr, f)


def dict_of_ECG():
    """
    Retourne un dictionnaire de pulsation extraites des ECG ECG_dict
    Ainsi que les correspondances avec la classe
    """
    ECG_dict = pickle.load(open('data/ECG_signals/ECG.pkl', 'rb'))
    corr_dict = pickle.load(open('data/ECG_signals/corr.pkl', 'rb'))

    return ECG_dict, corr_dict

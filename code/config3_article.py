from utils.classes import *

""" Signaux sinusoidaux et transformée de Fourier """

# ----- Config 3 -----

# fonctions
s = np.sin
c = np.cos

# constantes
pi = np.pi
np.random.seed(3)
random.seed(19)

# intervalle signal
x_min, x_max = 0, 3
abs_normal = np.linspace(x_min, x_max, ConstGraph_article.INPUT_SIZE_CONFIG_3)
abs_fft = fftfreq(ConstGraph_article.INPUT_SIZE_CONFIG_3, x_max)[:ConstGraph_article.INPUT_SIZE_CONFIG_3 // 2]

# Nombre de neurones
nb_neurons = 18


def dict_of_signal():
    """Retourne un dictionnaire de signaux"""

    def random_signal():
        signal = (-1) ** random.randint(1, 2) * random.uniform(0, 1) * s(abs_normal)
        common = random.randint(1, 3)
        for _ in range(11):
            signal += (-1) ** random.randint(1, 2) * random.uniform(0, 3) * c(
                pi * random.uniform(1, 3) * common * abs_normal) + (
                          -1) ** random.randint(1, 2) * random.uniform(0, 3) * s(
                pi * random.uniform(1, 3) * common * abs_normal)
        return signal

    fct = {}
    for i in range(nb_neurons):
        fct[i] = random_signal()
    return fct


# création des signaux
signaux = dict_of_signal()


def plot_dict_signal(abs, dict_y):
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


def dict_of_fft():
    """Retourne un dictionnaire de fft correspondant aux signaux"""
    fft_dict = {}
    for index_fft, s in enumerate(signaux.values()):
        fft_dict[index_fft] = 2.0 / ConstGraph_article.INPUT_SIZE_CONFIG_3 * np.abs(fft(s)[0:ConstGraph_article.INPUT_SIZE_CONFIG_3 // 2])
    return fft_dict


# création des FFT
FFT = dict_of_fft()


def print_cluster(G, display):
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
            fig.add_scatter(row=row_index, col=list(clusters.keys()).index(label)+1, x=abs, y=dict_y[neuron_index])
            row_index += 1
    fig.update_layout(
        paper_bgcolor=ConstPlotly.transparent_color,
        showlegend=False
    )
    plot(fig)


def train_model():
    # affichage des signaux brutes
    # plot_dict_signal(abs=abs_normal, dict_y=signaux)
    # Affichage des FFT
    # plot_dict_signal(abs=abs_fft, dict_y=FFT)
    # Création réseau et ajout neurones
    G = Graph()
    for i in range(nb_neurons):
        G.addNeuron(Neuron(vecteur=FFT[i]))
    # affichage de la config du réseau finale
    print_cluster(G, display=True)
    # Affichage des signaux brutes classés par cluster
    plot_signaux_par_cluster(G, abs=abs_normal, dict_y=signaux)


train_model()
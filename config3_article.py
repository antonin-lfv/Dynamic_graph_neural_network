from utils.classes import *

""" Version article """

# ----- Config 3 -----

# fonctions
s = np.sin
c = np.cos

# constantes
pi = np.pi
np.random.seed(90)
random.seed(90)

# intervalle signal
x_min, x_max = 0, 5
abs_normal = np.linspace(x_min, x_max, ConstGraph_article.INPUT_SIZE_CONFIG_3)
abs_fft = fftfreq(ConstGraph_article.INPUT_SIZE_CONFIG_3, x_max)[:ConstGraph_article.INPUT_SIZE_CONFIG_3 // 2]

# Nombre de neurones
nb_neurons = 15


def dict_of_signal():
    """Retourne un dictionnaire de signaux"""

    def random_signal():
        signal = (-1) ** random.randint(1, 2) * random.uniform(0, 1) * s(abs_normal)
        common = random.randint(1, 3)
        for _ in range(10):
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
    """Affiche un dictionnaire de signaux"""
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


# affichage des signaux brutes
# plot_dict_signal(abs=abs_normal, dict_y=signaux)


def dict_of_fft():
    """Retourne un dictionnaire de fft correspondant aux signaux"""
    fft_dict = {}
    for index_fft, s in enumerate(signaux.values()):
        fft_dict[index_fft] = 2.0 / ConstGraph_article.INPUT_SIZE_CONFIG_3 * np.abs(fft(s)[0:ConstGraph_article.INPUT_SIZE_CONFIG_3 // 2])
    return fft_dict


# création des FFT
FFT = dict_of_fft()

# Affichage des FFT
# plot_dict_signal(abs=abs_fft, dict_y=FFT)

# Création du réseau
G = Graph()
for i in range(nb_neurons):
    G.addNeuron(Neuron(vecteur=FFT[i]))

# affichage de la config du réseau
G.neurons

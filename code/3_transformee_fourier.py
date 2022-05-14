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

# intervalles signaux
x_min, x_max = 0, 3
abs_normal = np.linspace(x_min, x_max, ConstGraph_article.INPUT_SIZE_CONFIG_3)
abs_fft = fftfreq(ConstGraph_article.INPUT_SIZE_CONFIG_3, x_max)[:ConstGraph_article.INPUT_SIZE_CONFIG_3 // 2]

# Nombre de neurones
nb_neurons = 18

# création des signaux brutes
signaux = dict_of_signal(abscisse=abs_normal, nb_neurons=nb_neurons)

# création des FFT des signaux brutes
FFT = dict_of_fft(signaux=signaux)


def train_model(plot_brutes=False, plot_FFT=False, plot_brutes_par_cluster=True):
    # affichage des signaux brutes
    if plot_brutes:
        plot_dict_signal(abs=abs_normal, dict_y=signaux, signaux=signaux, nb_neurons=nb_neurons)
    # Affichage des FFT
    if plot_FFT:
        plot_dict_signal(abs=abs_fft, dict_y=FFT, signaux=signaux, nb_neurons=nb_neurons)
    # Création réseau et ajout neurones
    G = Graph()
    G.fit(FFT)
    # affichage de la config du réseau finale
    print_cluster(G, display=True)
    # Affichage des signaux brutes classés par cluster
    if plot_brutes_par_cluster:
        plot_signaux_par_cluster(G, abs=abs_normal, dict_y=signaux)


train_model(plot_brutes=False, plot_FFT=False, plot_brutes_par_cluster=True)

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

# création des signaux brutes :
# 1) signaux sinusoidaux aléatoires
nb_neurons = 18
signaux = dict_of_signal(abscisse=abs_normal, nb_neurons=nb_neurons)
# 2) chants d'oiseaux
# nb_neurons = 30
# signaux, corr = dict_of_birds()

# création des FFT des signaux brutes
FFT = dict_of_fft(signaux=signaux)


def main_sinusoid(plot_brutes=False, plot_FFT=False, plot_brutes_par_cluster=True):
    # affichage des signaux brutes
    if plot_brutes:
        plot_dict_signal(absc=abs_normal, dict_y=signaux, nb_neurons=nb_neurons)
    # Affichage des FFT
    if plot_FFT:
        plot_dict_signal(absc=abs_fft, dict_y=FFT, nb_neurons=nb_neurons)
    # Création réseau et ajout neurones
    G = Graph()
    G.fit(FFT)
    # affichage de la config du réseau finale
    print_cluster(G, display=True)
    # Affichage des signaux brutes classés par cluster
    if plot_brutes_par_cluster:
        plot_signaux_par_cluster(G, absc=abs_normal, dict_y=signaux)


# main(plot_brutes=False, plot_FFT=False, plot_brutes_par_cluster=True)


def main_birds(plot_brutes=False, plot_FFT=False, plot_brutes_par_cluster=True):
    # affichage des signaux brutes
    if plot_brutes:
        plot_dict_signal(dict_y=signaux, nb_neurons=nb_neurons)
    # Affichage des FFT
    if plot_FFT:
        plot_dict_signal(dict_y=FFT, nb_neurons=nb_neurons)
    # Création réseau et ajout neurones
    G = Graph()
    G.fit(FFT)
    # affichage de la config du réseau finale
    print_cluster(G, display=True)
    # Affichage des signaux brutes classés par cluster
    if plot_brutes_par_cluster:
        plot_signaux_par_cluster(G, dict_y=signaux)


# main(plot_brutes=False, plot_FFT=False, plot_brutes_par_cluster=True)

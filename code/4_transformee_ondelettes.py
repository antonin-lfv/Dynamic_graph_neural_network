from utils.classes import *

""" Signaux sinusoidaux et ondelettes """

# ----- Config 4 -----

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
abs_wv = abs_normal

# Nombre de neurones
nb_neurons = 18

# création des signaux brutes
signaux = dict_of_signal(abs_normal, nb_neurons)


def test_affiche_decomp_ondelettes(signal):
    """Plot le signal, et sa décomposition en ondelettes
    @:param signal: le signal
    """
    (cA, cD) = pywt.dwt(signal, 'db20')
    fig = make_subplots(rows=3, cols=1, subplot_titles=["origin", "cA", "cD"], shared_xaxes=True)
    fig.add_scatter(x=abs_wv, y=signaux[0], row=1, col=1)
    fig.add_scatter(x=abs_wv, y=cA, row=2, col=1)
    fig.add_scatter(x=abs_wv, y=cD, row=3, col=1)
    plot(fig)


test_affiche_decomp_ondelettes(signaux[7])


# création des WV
WV = dict_of_wv(signaux=signaux)


def main(plot_brutes=False, plot_FFT=False, plot_brutes_par_cluster=True):
    # affichage des signaux brutes
    if plot_brutes:
        plot_dict_signal(abs=abs_normal, dict_y=signaux, signaux=signaux, nb_neurons=nb_neurons)
    # Affichage des FFT
    if plot_FFT:
        plot_dict_signal(abs=abs_wv, dict_y=WV, signaux=signaux, nb_neurons=nb_neurons)
    # Création réseau et ajout neurones
    G = Graph()
    G.fit(X=WV)
    # affichage de la config du réseau finale
    print_cluster(G, display=True)
    # Affichage des signaux brutes classés par cluster
    if plot_brutes_par_cluster:
        plot_signaux_par_cluster(G, abs=abs_normal, dict_y=signaux)


main(plot_brutes=False, plot_FFT=False, plot_brutes_par_cluster=True)

from utils.classes import *

""" Chants d'oiseaux """

# chants d'oiseaux

config = {
    "bv": 0.40,
    "bc": 0.40,
    "bl": 0.40,
    "ar": 60,
    "an": 68
}

# create_dict_of_birds()  # one time
signaux, corr = dict_of_birds()
# prendre que les 2 premières classes
nb_neurons = 10
signaux = dict(itertools.islice(signaux.items(), 10))


def main_birds(plot_brutes=False, plot_brutes_par_cluster=True):
    # affichage des signaux brutes
    if plot_brutes:
        plot_dict_signal(dict_y=signaux, nb_neurons=nb_neurons)
    # Création réseau et ajout neurones
    G = Graph(config=config, fct_distance=distance_neurons_DTW)  # on utilise la distance DTW
    G.fit(signaux)
    # affichage de la config du réseau finale
    print_cluster(G, display=True)
    # Affichage des signaux brutes classés par cluster
    if plot_brutes_par_cluster:
        plot_signaux_par_cluster(G, dict_y=signaux)


main_birds(plot_brutes=False, plot_brutes_par_cluster=True)

from utils.classes import *

""" signaux sinusoidaux et transformée en ondelettes """

# ----- Config -----

config = {
    "INPUT_SIZE": 250,
    "bv": 0.30,
    "bc": 0.20,
    "bl": 0.20,
    "ar": 2.1,
    "an": 1.2
}

# constantes
pi = np.pi
np.random.seed(3)
random.seed(19)

# intervalles signaux
x_min, x_max = 0, 3
abs_normal = np.linspace(x_min, x_max, config["INPUT_SIZE"])

# création des signaux brutes :
nb_neurons = 18
signaux = dict_of_signal(abscisse=abs_normal, nb_neurons=nb_neurons)

# Ondelettes
Wavelet = dict_of_wavelet(signaux, scale=30)

if __name__ == '__main__':
    G = Graph(config=config, suppr_neuron=False)
    G.fit(Wavelet, print_progress=False)
    G.print_cluster(display=True)
    G.graphInfo()
    print(f"Nombre de clusters crées : {len(G.print_cluster(display=False).keys())}")
    # G.get_size_of_links(plotly_fig=True)
    plot_signaux_par_cluster(G, dict_y=signaux, sign_min_per_cluster=2)

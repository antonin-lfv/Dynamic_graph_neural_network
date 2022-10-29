from utils.classes import *

""" ECG et transformée en ondelettes """

# ----- Config -----

config = {
    "INPUT_SIZE": None,
    "bv": 3,
    "bc": 2,
    "bl": 3,
    "ar": 0.018,
    "an": 0.013
}

# ----- ECG -----
# create_dict_of_ECG()  # one time
ECG, corr = dict_of_ECG()

# création des FFT des signaux
config["INPUT_SIZE"] = len(ECG[0])
FFT = dict_of_fft(signaux=ECG, taille_signaux=config["INPUT_SIZE"])


def main_sinusoid(plot_brutes=False, plot_FFT=False, plot_brutes_par_cluster=False):
    # affichage des signaux brutes
    if plot_brutes:
        plot_dict_signal(dict_y=ECG, nb_neurons=len(ECG))
    # Affichage des FFT
    if plot_FFT:
        plot_dict_signal(dict_y=FFT, nb_neurons=len(ECG))
    # Création réseau et ajout neurones
    G = Graph(config=config, suppr_neuron=True)
    G.fit(FFT, print_progress=False)
    # affichage de la config du réseau finale
    print_cluster(G, display=True)
    print(f"Nombre de neurones supprimés : {len(ECG) - len(G.neurons)}")
    # Affichage des signaux brutes classés par cluster
    if plot_brutes_par_cluster:
        plot_signaux_par_cluster(G, dict_y=ECG, sign_min_per_cluster=2)
    return G


if __name__ == "__main__":
    Graphe = main_sinusoid(plot_brutes=False, plot_FFT=False, plot_brutes_par_cluster=True)
    # Graphe.get_size_of_links(plotly_fig=False)

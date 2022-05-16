from utils.classes import *

""" Chants d'oiseaux et transformée de Fourier """

# chants d'oiseaux
nb_neurons = 30
signaux, corr = dict_of_birds()

# Affichage sons
path1 = "data/Birdsong/Bird2/Wave/1.wav"
y1 = read(path1)
fig = go.Figure()
fig.add_scatter(y=y1)
fig.update_layout(
    paper_bgcolor=ConstPlotly.transparent_color,
    showlegend=False
)
plot(fig)

# distance, path = fastdtw(y1, y2[:84562], dist=euclidean)

FFT = None


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
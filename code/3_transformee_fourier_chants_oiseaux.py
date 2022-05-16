from utils.classes import *

""" Chants d'oiseaux et transformée de Fourier """

# chants d'oiseaux
nb_neurons = 30
# signaux, corr = dict_of_birds()

# Affichage sons
path1 = "data/Birdsong/Bird1/Wave/3.wav"
path2 = "data/Birdsong/Bird0/Wave/0.wav"
y1 = read(path1)
y2 = read(path2)
fig = go.Figure()
fig.add_scatter(y=y1/np.linalg.norm(y1))
fig.add_scatter(y=y2/np.linalg.norm(y2))
plot(fig)
signaux = {0: y1}

# création des FFT des chants
FFT = dict_of_fft(signaux=signaux, taille_signaux=len(y1))
print(len(y1))
print(len(FFT[0]))

fig = go.Figure()
fig.add_scatter(y=FFT[0])
fig.update_layout(
    paper_bgcolor=ConstPlotly.transparent_color,
    showlegend=False
)
plot(fig)

distance, path = fastdtw(y1, y2[:84562], dist=euclidean)


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
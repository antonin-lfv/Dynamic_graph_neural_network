from utils.classes import *

""" Chants d'oiseaux et transformée de Fourier """

# chants d'oiseaux
nb_neurons = 30
# signaux, corr = dict_of_birds()

# Affichage sons
path = "data/Birdsong/Bird1/Wave/3.wav"
fig = go.Figure()
y = read(path)
fig.add_scatter(y=y)
fig.update_layout(
    paper_bgcolor=ConstPlotly.transparent_color,
    showlegend=False
)
plot(fig)

signaux = {0: y}

# création des FFT des chants
FFT = dict_of_fft(signaux=signaux, taille_signaux=len(y))
print(len(y))
print(len(FFT[0]))

fig = go.Figure()
fig.add_scatter(y=FFT[0])
fig.update_layout(
    paper_bgcolor=ConstPlotly.transparent_color,
    showlegend=False
)
plot(fig)


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

from utils.classes import *

""" Signaux sinusoidaux """

# ----- Config -----
# Avec 9 neurones

config = {
    "INPUT_SIZE": 300,
    "bv": 0.30,
    "bc": 0.20,
    "bl": 0.20,
    "ar": 2.1,
    "an": 1.2
}

# fonctions
s = np.sin
c = np.cos

# constantes
pi = np.pi
np.random.seed(1)
random.seed(1)

# intervalle signal
inter = np.linspace(0, 5, config["INPUT_SIZE"])

# Création des signaux brutes
signaux = dict_of_signal(inter, 9)

# Réseau
G = Graph(config=config)
G.fit(X=signaux)


def affichage_signaux():
    """Affichage de tous les signaux"""
    fig = make_subplots(cols=3, rows=3, shared_xaxes=True, subplot_titles=[f"Neurone {i}" for i in [0, 3, 6, 1, 4, 7, 2, 5, 8]])
    for row_index, sign in enumerate(signaux.values()):
        fig.add_scatter(y=sign, x=inter, row=row_index % 3 + 1, col=1 if row_index < 3 else 2 if row_index < 6 else 3)
    fig.update_layout(
        paper_bgcolor=ConstPlotly.transparent_color,
        showlegend=False
    )
    plot(fig)


# Affichage des signaux des neurones
# affichage_signaux()

# Affichage des signaux par clusters
plot_signaux_par_cluster(G, signaux, absc=inter, sign_min_per_cluster=2)

# affichage de la config du réseau
G.neurons

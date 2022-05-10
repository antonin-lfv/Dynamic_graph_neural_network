from utils.classes import *

""" Version article """

# ----- Config 2 -----

# fonctions
s = np.sin
c = np.cos

# constantes
pi = np.pi
np.random.seed(1)
random.seed(1)

# intervalle signal
inter = np.linspace(0, 5, ConstGraph_article.INPUT_SIZE_CONFIG_2)


def dict_of_signal():
    def random_signal():
        signal = (-1)**random.randint(1, 2)*random.uniform(0, 1)*s(inter)
        common = random.randint(1, 3)
        for i in range(15):
            signal += (-1)**random.randint(1, 2)*random.uniform(0, 1)*c(pi*common*inter)+(-1)**random.randint(1, 2)*random.uniform(0, 1)*s(pi*common*inter)
        return signal
    fct = {}
    for i in range(9):
        fct[i] = random_signal()
    return fct


signaux = dict_of_signal()

# Réseau

G = Graph()
for i in range(9):
    G.addNeuron(Neuron(vecteur=signaux[i]))


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
affichage_signaux()

# affichage de la config du réseau
# G.neurons

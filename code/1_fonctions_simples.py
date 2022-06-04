from utils.classes import *

""" Fonctions classiques """

# ----- Config -----

config = {
    "INPUT_SIZE": 100,
    "bv": 0.10,
    "bc": 0.10,
    "bl": 0.50,
    "ar": 150,
    "an": 100
}

# fonctions
f = np.cos
g = np.sqrt

type_1 = {
    1: 1 * f(np.linspace(0, 5, config["INPUT_SIZE"])),
    2: 2 * f(np.linspace(0, 5, config["INPUT_SIZE"])),
    3: 3 * f(np.linspace(0, 5, config["INPUT_SIZE"])),
    4: 4 * f(np.linspace(0, 5, config["INPUT_SIZE"])),
    5: 5 * f(np.linspace(0, 5, config["INPUT_SIZE"])),
}

type_2 = {
    1: 14 * g(np.linspace(0, 5, config["INPUT_SIZE"])),
    2: 15 * g(np.linspace(0, 5, config["INPUT_SIZE"])),
    3: 16 * g(np.linspace(0, 5, config["INPUT_SIZE"])),
    4: 17 * g(np.linspace(0, 5, config["INPUT_SIZE"])),
    5: 18 * g(np.linspace(0, 5, config["INPUT_SIZE"])),
}

# Groupe de neurones 1 : 0, 3, 4, 6, 7
# Groupe de neurones 2 : 1, 2, 5, 8, 9

G = Graph(config=config)
# Ajout à la main (Utiliser la méthode fit à la place)
G.addNeuron(Neuron(vecteur=type_1[1], config=config))  # 0
G.addNeuron(Neuron(vecteur=type_2[2], config=config))  # 1
G.addNeuron(Neuron(vecteur=type_2[3], config=config))  # 2
G.addNeuron(Neuron(vecteur=type_1[2], config=config))  # 3
G.addNeuron(Neuron(vecteur=type_1[4], config=config))  # 4
G.addNeuron(Neuron(vecteur=type_2[5], config=config))  # 5
G.addNeuron(Neuron(vecteur=type_1[3], config=config))  # 6
G.addNeuron(Neuron(vecteur=type_1[5], config=config))  # 7
G.addNeuron(Neuron(vecteur=type_2[1], config=config))  # 8
G.addNeuron(Neuron(vecteur=type_2[4], config=config))  # 9


# Affichage des vecteurs des neurones de cette config
def plot_neurons_config_1_article():
    L = list(type_1.values()) + list(type_2.values())
    index = [0, 3, 6, 4, 7, 8, 1, 2, 9, 5]
    fig = go.Figure()
    for l, j in zip(L, index):
        fig.add_scatter(x=np.linspace(0, 5, config["INPUT_SIZE"]), y=l, name=f"index : {j}")
    fig.update_layout(
        paper_bgcolor=ConstPlotly.transparent_color,
        plot_bgcolor=ConstPlotly.transparent_color,
    )
    plot(fig)


# affichage des vecteurs des neurones
# plot_neurons_config_1_article()

# affichage de la config du réseau
G.neurons

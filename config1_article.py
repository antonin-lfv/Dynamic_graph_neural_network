from utils.classes import *

""" Version article """

# ----- Config 1 -----

# fonctions
f = np.cos
g = np.sqrt

type_1 = {
    1: 1 * f(np.linspace(0, 5, ConstGraph_article.INPUT_SIZE_CONFIG_1)),
    2: 2 * f(np.linspace(0, 5, ConstGraph_article.INPUT_SIZE_CONFIG_1)),
    3: 3 * f(np.linspace(0, 5, ConstGraph_article.INPUT_SIZE_CONFIG_1)),
    4: 4 * f(np.linspace(0, 5, ConstGraph_article.INPUT_SIZE_CONFIG_1)),
    5: 5 * f(np.linspace(0, 5, ConstGraph_article.INPUT_SIZE_CONFIG_1)),
}

type_2 = {
    1: 14 * g(np.linspace(0, 5, ConstGraph_article.INPUT_SIZE_CONFIG_1)),
    2: 15 * g(np.linspace(0, 5, ConstGraph_article.INPUT_SIZE_CONFIG_1)),
    3: 16 * g(np.linspace(0, 5, ConstGraph_article.INPUT_SIZE_CONFIG_1)),
    4: 17 * g(np.linspace(0, 5, ConstGraph_article.INPUT_SIZE_CONFIG_1)),
    5: 18 * g(np.linspace(0, 5, ConstGraph_article.INPUT_SIZE_CONFIG_1)),
}

# Groupe de neurones 1 : 0, 3, 4, 6, 7
# Groupe de neurones 2 : 1, 2, 5, 8, 9

G = Graph()
G.addNeuron(Neuron(vecteur=type_1[1]))  # 0
G.addNeuron(Neuron(vecteur=type_2[2]))  # 1
G.addNeuron(Neuron(vecteur=type_2[3]))  # 2
G.addNeuron(Neuron(vecteur=type_1[2]))  # 3
G.addNeuron(Neuron(vecteur=type_1[4]))  # 4
G.addNeuron(Neuron(vecteur=type_2[5]))  # 5
G.addNeuron(Neuron(vecteur=type_1[3]))  # 6
G.addNeuron(Neuron(vecteur=type_1[5]))  # 7
G.addNeuron(Neuron(vecteur=type_2[1]))  # 8
G.addNeuron(Neuron(vecteur=type_2[4]))  # 9


# Affichage des vecteurs des neurones de cette config
def plot_neurons_config_1_article():
    L = list(type_1.values()) + list(type_2.values())
    index = [0, 3, 6, 4, 7, 8, 1, 2, 9, 5]
    fig = go.Figure()
    for l, j in zip(L, index):
        fig.add_scatter(x=np.linspace(0, 5, ConstGraph_article.INPUT_SIZE_CONFIG_1), y=l, name=f"index : {j}")
    fig.update_layout(
        paper_bgcolor=ConstPlotly.transparent_color,
        plot_bgcolor=ConstPlotly.transparent_color,
    )
    plot(fig)


# plot_neurons_config_1_article()

G.neurons
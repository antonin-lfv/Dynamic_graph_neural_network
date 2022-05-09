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
            signal += (-1)**random.randint(1, 2)*random.uniform(0, 1)*s(pi*common*inter)
        return signal
    fct = {}
    for i in range(9):
        fct[i] = random_signal()
    return fct


signaux = dict_of_signal()


def affichage_signaux():
    fig = make_subplots(cols=1, rows=len(signaux), shared_xaxes=True)
    for row_index, sign in zip([i for i in range(1, len(signaux)+1)], signaux.values()):
        fig.add_scatter(y=sign, x=inter, row=row_index, col=1)
    fig.update_layout(
        paper_bgcolor=ConstPlotly.transparent_color,
    )
    plot(fig)


# Réseau

G = Graph()
G.addNeuron(Neuron(vecteur=signaux[0]))  # 0
G.addNeuron(Neuron(vecteur=signaux[1]))  # 1
G.addNeuron(Neuron(vecteur=signaux[2]))  # 2
G.addNeuron(Neuron(vecteur=signaux[3]))  # 3
G.addNeuron(Neuron(vecteur=signaux[4]))  # 4
G.addNeuron(Neuron(vecteur=signaux[5]))  # 5
G.addNeuron(Neuron(vecteur=signaux[6]))  # 6
G.addNeuron(Neuron(vecteur=signaux[7]))  # 7
G.addNeuron(Neuron(vecteur=signaux[8]))  # 8

# Affichage des signaux des neurones
# affichage_signaux()

# affichage de la config du réseau
G.neurons

for i in range(len(G.neurons)):
    print(distance_neurons(G.neurons[0].vecteur, G.neurons[i].vecteur))

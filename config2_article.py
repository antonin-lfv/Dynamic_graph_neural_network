from utils.classes import *

""" Version article """

# ----- Config 2 -----

# fonctions
s = np.sin
c = np.cos

# constantes
pi = np.pi
np.random.seed(1)

# intervalle signal
inter = np.linspace(0, 10, ConstGraph_article.INPUT_SIZE_CONFIG_2)


def dict_of_signal():
    def random_signal():
        signal = (-1)**random.randint(1, 2)*random.uniform(0, 1)*s(inter)
        for i in range(5):
            signal += (-1)**random.randint(1, 2)*random.uniform(0, 1)*s(pi*i*inter)
        return signal
    fct = {}
    for i in range(9):
        fct[i] = random_signal()
    return fct


signaux = dict_of_signal()


def affichage_signaux():
    fig = go.Figure()
    for sign in signaux.values():
        fig.add_scatter(y=sign, x=inter)
    plot(fig)

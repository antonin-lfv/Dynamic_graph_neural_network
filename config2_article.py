from utils.classes import *

""" Version article """

# ----- Config 2 -----

# fonctions
s = np.sin
c = np.cos

pi = np.pi


def dict_of_signal():
    def random_signal():
        signal = random.uniform(0, 1)*s(np.linspace(0, 10, ConstGraph_article.INPUT_SIZE_CONFIG_2))
        for i in range(5):
            signal += (-1)**random.randint(1, 2)*random.uniform(0, 1)*s(pi*i*np.linspace(0, 10, ConstGraph_article.INPUT_SIZE_CONFIG_2))
        return signal
    fct = {}
    for i in range(16):
        fct[i] = random_signal()
    return fct


signaux = dict_of_signal()

from utils.classes import *

""" Signaux sinusoidaux et transformée de Fourier """

# ----- Config -----

config = {
    "INPUT_SIZE": 250,
    "bv": 0.30,
    "bc": 0.20,
    "bl": 0.20,
    "ar": 25,
    "an": 6.5
}

# abscisses signaux
x_min, x_max = 0, 3
abs_normal = np.linspace(x_min, x_max, config["INPUT_SIZE"])
# création des signaux brutes:
nb_neurons = 15
brutes = dict_of_signal(abscisse=abs_normal,
                        nb_neurons=nb_neurons)



if __name__ == '__main__':
    ...

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

# abscisses signaux (taille fixe car FFT)
x_min, x_max = 0, 3
abs_normal = np.linspace(x_min, x_max, config["INPUT_SIZE"])
abs_fft = fftfreq(config["INPUT_SIZE"], x_max)[:config["INPUT_SIZE"] // 2]

"""
Ici, on va appliquer la méthode décrite dans le ReadMe dans la partie Utilisation du modèle

1. On applique le modèle sur toutes nos données
2. On regarde le cluster avec le plus de neurone, et on relance le modèle sur toutes les données sauf ce cluster
3. On revient à l'étape 2

On peut arrêter le processus quand il ne reste plus qu'une seule donnée, ou alors on peut fixer un nombre de tour. 
Ce processus va réduire l'échelle petit à petit, et on pourra ainsi distinguer des groupes de données qui semblaient 
similaires en regardant l'ensemble des données.
"""


if __name__ == '__main__':
    # création des signaux brutes:
    nb_neurons = 15
    brutes = dict_of_signal(abscisse=abs_normal,
                            nb_neurons=nb_neurons)

    model = ClassificationDNN(raw_data=brutes,
                              abscisse=abs_normal,
                              config=config,
                              nb_iteration=2)
    model.fit()
    model.showResult()

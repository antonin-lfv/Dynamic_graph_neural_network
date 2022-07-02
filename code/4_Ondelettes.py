from utils.classes import *

""" Signaux sinusoidaux et transformée de Fourier """

# ----- Config -----

config = {
    "INPUT_SIZE": 250,
    "bv": 0.30,
    "bc": 0.20,
    "bl": 0.20,
    "ar": 25,
    "an": 150
}

# ===== Signaux sinusoïdaux
x_min, x_max = 0, 3
abs_normal = np.linspace(x_min, x_max, config["INPUT_SIZE"])
nb_neurons = 20
brutes = dict_of_signal(abscisse=abs_normal,
                        nb_neurons=nb_neurons)

# ===== ECG
numpy_from_matlab('data/ECG_signals/1 NSR/100m (0).mat')
get_file_in_folder("data/ECG_signals/1 NSR")
Wavelet = dict_of_wavelet(brutes)


if __name__ == '__main__':
    G = Graph(config=config)
    G.fit(Wavelet, print_progress=False)
    # G.print_cluster(display=True)
    plot_signaux_par_cluster(G, absc=abs_normal, dict_y=brutes, sign_min_per_cluster=2)

from utils.classes import *

""" ECG et transformée en ondelettes """

# ----- Config -----

config = {
    "INPUT_SIZE": None,
    "bv": 5,
    "bc": 6,
    "bl": 2,
    "ar": 0.27,
    "an": 0.09
}

# ----- ECG -----
create_dict_of_ECG()  # one time
signaux, corr = dict_of_ECG()

ECG, count_index = {}, 0
prefixe_path = "data/ECG_signals"
folders_name = ['data/ECG_signals/1 NSR',
                'data/ECG_signals/2 APB',
                ]

"""All folders
'data/ECG_signals/1 NSR',
'data/ECG_signals/2 APB',
'data/ECG_signals/3 AFL',
'data/ECG_signals/4 AFIB',
'data/ECG_signals/5 SVTA',
'data/ECG_signals/6 WPW',
'data/ECG_signals/7 PVC',
'data/ECG_signals/8 Bigeminy',
'data/ECG_signals/9 Trigeminy',
'data/ECG_signals/10 VT',
'data/ECG_signals/11 IVR',
'data/ECG_signals/12 VFL',
'data/ECG_signals/13 Fusion',
'data/ECG_signals/14 LBBBB',
'data/ECG_signals/15 RBBBB',
'data/ECG_signals/16 SDHB',
'data/ECG_signals/17 PR'
"""

# On garde seulement n data de chacun des k types ! Soit k * n données
k = 10
for data_folder in folders_name:
    folders = get_file_in_folder(data_folder)
    compt = 0
    for sample_data_path in folders:
        if compt < k:
            ECG[count_index] = numpy_from_matlab(sample_data_path)
            count_index += 1
            compt += 1
        else:
            break


# création des FFT des signaux
config["INPUT_SIZE"] = len(ECG[0])
FFT = dict_of_fft(signaux=ECG, taille_signaux=config["INPUT_SIZE"])


def main_sinusoid(plot_brutes=False, plot_FFT=False, plot_brutes_par_cluster=True):
    # affichage des signaux brutes
    if plot_brutes:
        plot_dict_signal(dict_y=ECG, nb_neurons=len(ECG))
    # Affichage des FFT
    if plot_FFT:
        plot_dict_signal(dict_y=FFT, nb_neurons=len(ECG))
    # Création réseau et ajout neurones
    G = Graph(config=config, suppr_neuron=True)
    G.fit(FFT, print_progress=False)
    # affichage de la config du réseau finale
    print_cluster(G, display=True)
    print(f"Nombre de neurones supprimés : {len(ECG) - len(G.neurons)}")
    # Affichage des signaux brutes classés par cluster
    if plot_brutes_par_cluster:
        plot_signaux_par_cluster(G, dict_y=ECG, sign_min_per_cluster=2)
    return G


if __name__ == "__main__":
    Graphe = main_sinusoid(plot_brutes=False, plot_FFT=False, plot_brutes_par_cluster=False)
    Graphe.get_size_of_links(plotly_fig=False)

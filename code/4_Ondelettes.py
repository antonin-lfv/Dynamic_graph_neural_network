from utils.classes import *

""" ECG et transformée en ondelettes """

# ----- Config -----

config = {
    "INPUT_SIZE": 250,
    "bv": 0.30,
    "bc": 0.20,
    "bl": 0.20,
    "ar": 14000,
    "an": 9000
}

# ----- ECG -----
ECG, count_index = {}, 0
prefixe_path = "data/ECG_signals"
folders_name = ['data/ECG_signals/1 NSR',
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
                ]

# On garde seulement 10 data de chaque type ! Soit 17 * 10 = 170 données
for data_folder in folders_name:
    folders = get_file_in_folder(data_folder)
    compt = 0
    for sample_data_path in folders:
        if compt < 5:
            ECG[count_index] = numpy_from_matlab(sample_data_path)
            count_index += 1
            compt += 1
        else:
            break

# création des ondelettes
Wavelet = dict_of_wavelet(ECG)
# Wavelet = shuffle_dict(Wavelet)


if __name__ == '__main__':
    G = Graph(config=config, suppr_neuron=True)
    G.fit(Wavelet, print_progress=False, use_existing_index=True)
    G.print_cluster(display=True)
    G.graphInfo()
    print(f"Nombre de clusters crées : {len(G.print_cluster(display=False).keys())}")

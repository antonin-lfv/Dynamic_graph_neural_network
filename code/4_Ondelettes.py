from utils.classes import *

""" ECG et transformée en ondelettes """

# ----- Config -----

config = {
    "INPUT_SIZE": 250,
    "bv": 0.30,
    "bc": 0.20,
    "bl": 0.20,
    "ar": 350,
    "an": 9000
}

# ----- ECG -----
ECG, count_index = {}, 0
prefixe_path = "data/ECG_signals"
folders_name = get_file_in_folder(prefixe_path)

# On garde seulement 10 data de chaque type ! Soit 17 * 10 = 170 données
for data_folder in folders_name:
    folders = get_file_in_folder(data_folder)
    compt = 0
    for sample_data_path in folders:
        if compt < 10:
            ECG[count_index] = numpy_from_matlab(sample_data_path)
            count_index += 1
            compt += 1
        else:
            break

# création des ondelettes
Wavelet = dict_of_wavelet(ECG)
Wavelet = shuffle_dict(Wavelet)

if __name__ == '__main__':
    G = Graph(config=config, suppr_neuron=True)
    G.fit(Wavelet, print_progress=False)
    G.print_cluster(display=True)
    G.graphInfo()
    print(f"Nombre de clusters crées : {len(G.print_cluster(display=False).keys())}")

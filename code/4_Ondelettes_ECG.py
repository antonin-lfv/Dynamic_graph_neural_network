from utils.classes import *

""" ECG et transformée en ondelettes """

# ----- Config -----

config = {
    "INPUT_SIZE": None,
    "bv": 3,
    "bc": 2,
    "bl": 3,
    "ar": 0.3,
    "an": 0.15
}

# ----- ECG -----
# create_dict_of_ECG()  # one time
ECG, corr = dict_of_ECG()

# création des ondelettes
Wavelet = dict_of_wavelet(ECG)

if __name__ == '__main__':
    G = Graph(config=config, suppr_neuron=False)
    G.fit(Wavelet, print_progress=False)
    G.print_cluster(display=True)
    G.graphInfo()
    print(f"Nombre de clusters crées : {len(G.print_cluster(display=False).keys())}")
    # G.get_size_of_links(plotly_fig=True)

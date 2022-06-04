from utils.classes import *

"""On prendra une taille de signaux égale à 2041 pour les signaux sinusoidaux car c'est la taille moyenne des 
syllabes des chants d'oiseaux """


def temps_exec_birds_DTW(nb_signaux):
    # config
    config = {
        "bv": 0.40,
        "bc": 0.40,
        "bl": 0.40,
        "ar": 60,
        "an": 68
    }
    # syllabes
    signaux, corr = dict_of_birds()
    # prendre que les 2 premières classes
    nb_neurons = nb_signaux
    signaux = dict(itertools.islice(signaux.items(), nb_neurons))

    def main_birds(plot_brutes=False, plot_brutes_par_cluster=True):
        # affichage des signaux brutes
        if plot_brutes:
            plot_dict_signal(dict_y=signaux, nb_neurons=nb_neurons)
        # Création réseau et ajout neurones
        G = Graph(config=config, fct_distance=distance_neurons_DTW)  # on utilise la distance DTW
        G.fit(signaux)
        # affichage de la config du réseau finale
        # print_cluster(G, display=True)
        # Affichage des signaux brutes classés par cluster
        if plot_brutes_par_cluster:
            plot_signaux_par_cluster(G, dict_y=signaux)

    start_time = time.time()
    main_birds(plot_brutes=False, plot_brutes_par_cluster=False)
    return time.time() - start_time


def temps_exec_sinus_fourier(nb_signaux):
    # config
    config = {
        "INPUT_SIZE": 250,
        "bv": 0.30,
        "bc": 0.20,
        "bl": 0.20,
        "ar": 30,
        "an": 6.5
    }
    # intervalles signaux
    x_min, x_max = 0, 3
    abs_normal = np.linspace(x_min, x_max, 2041)
    abs_fft = fftfreq(2041, x_max)[:2041 // 2]

    # création des signaux brutes :
    nb_neurons = nb_signaux
    signaux = dict_of_signal(abscisse=abs_normal, nb_neurons=nb_neurons)

    # création des FFT des signaux brutes
    FFT = dict_of_fft(signaux=signaux, taille_signaux=2041)

    def main_sinusoid(plot_brutes=False, plot_FFT=False, plot_brutes_par_cluster=True):
        # affichage des signaux brutes
        if plot_brutes:
            plot_dict_signal(absc=abs_normal, dict_y=signaux, nb_neurons=nb_neurons)
        # Affichage des FFT
        if plot_FFT:
            plot_dict_signal(absc=abs_fft, dict_y=FFT, nb_neurons=nb_neurons)
        # Création réseau et ajout neurones
        G = Graph(config=config, fct_distance=None)
        G.fit(FFT)
        # affichage de la config du réseau finale
        # print_cluster(G, display=True)
        # Affichage des signaux brutes classés par cluster
        if plot_brutes_par_cluster:
            plot_signaux_par_cluster(G, absc=abs_normal, dict_y=signaux)

    start_time = time.time()
    main_sinusoid(plot_brutes=False, plot_FFT=False, plot_brutes_par_cluster=False)
    return time.time() - start_time


"""On fera le test sur plusieurs nombre de neurones, de 1 à 15"""
fig = go.Figure()
# temps_fourier = [temps_exec_sinus_fourier(i) for i in range(1, 15)]
temps_fourier = [0.00015592575073242188, 0.8240890502929688, 0.0022690296173095703, 0.0030930042266845703,
                 0.0017189979553222656, 0.0055768489837646484, 0.006134986877441406, 0.009235143661499023,
                 0.008621931076049805,
                 0.011160135269165039, 0.01581120491027832, 0.016182899475097656, 0.022082090377807617,
                 0.03059673309326172]
# temps_dtw = [temps_exec_birds_DTW(i) for i in range(1, 15)]
temps_dtw = [0.00012493133544921875, 5.3734941482543945, 8.1061110496521, 16.346383094787598, 19.14073896408081,
             20.229499101638794, 26.18322205543518, 41.730777740478516, 40.03119397163391, 44.81838893890381,
             111.86257886886597, 235.718444108963, 333.8916928768158, 442.79283618927]
fig.add_scatter(y=temps_fourier, mode="lines", line=dict(color="red"), name="Méthode avec transformée Fourier")
fig.add_scatter(y=temps_dtw, mode="lines", line=dict(color="blue"), name="Méthode avec DTW")
fig.update_xaxes(title="Nombre de neurones dans le réseau")
fig.update_yaxes(title="Temps d'exécution (sec)")
plot(fig)

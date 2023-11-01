<p align="center">
	<img src="https://user-images.githubusercontent.com/63207451/114284722-45901b80-9a52-11eb-8a0c-e99fc8681436.gif" height="80" width="140" alt="">
	</p>

<h1 align="center">Dynamic graph neural network</h1>

<br>

<p align="center">
<a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.10-2ea44f" alt="Python - 3.10"></a>
<a href="https://en.wikipedia.org/wiki/Types_of_artificial_neural_networks#Dynamic"><img src="https://img.shields.io/badge/Dynamic-Neural_Network-018291" alt="Dynamic - Neural Network"></a>

</p>

<br>

<p align="center">
Les réseaux de neurones dynamiques sont une branche peu développée du Deep Learning, qui repose sur un principe simple, l'architecture du réseau est dynamique. Cela a plusieurs avantages, notamment le fait que le réseau est en perpetuel apprentissage, et qu'il peut changer sa structure et le routage des informations en fonction des données, ce qui le rend très flexible. Le dynamisme s'opère au niveau de la profondeur du réseau mais aussi sur sa largeur (dans le cas d'un réseau en couches comme dans le modèle DAN2). Ici, on va encore un peu plus loin dans ce concept de réseau de neurones dynamiques, car on va utiliser une structure de graphe, c'est à dire sans organisation en couches comme les modèles classiques. <br>
	</p>

<p align="center">
Concernant ce projet, il a pour objectif de tester le pouvoir classificateur d'un réseau de neurones dynamiques en graphe décrit par l'article scientifique dont le lien est ci-dessous (le deuxième). L'article n'évoque qu'une partie mathématique et quelques voix pour la mise en place du modèle. Ainsi, La première étape de ce projet sera d'implémenter la structure du graphe ainsi que les méthodes associées telles qu'elles sont décrites dans cet article, puis, en fonction des resultats, d'améliorer le modèle. On verra que l'utilisation de la transformée de Fourier et la transformée en ondelettes sera nécessaire. Ensuite, une partie graphique sera implémentée avec la librairie Plotly qui servira à s'assurer de la bonne mise en place des premières méthodes (uniquement sur la première partie de l'implémentation). Concernant les phases d'expérimentation, on tentera pour commencer de classer plusieurs types de fonctions classiques, puis nous poursuivrons sur une classification de signaux sinusoïdaux, tout cela sans pré traitement des données, et nous tenterons de classer des chants d'oiseaux avec une autre méthode de calcul de distance. Ensuite, nous testerons la modification de nos données d'entrées avec une transformée de Fourier, ou encore une transformées en Ondelettes sur des électrocardiogrammes et des signaux sinusoïdaux. Dans ce repository, une première partie sera consacrée à l'aspect mathématique du modèle, pour mieux comprendre son fonctionnement. Puis sera expliqué l'implémentation avec Python avec les différents tests et résultats.
	</p>

<br>



<br>

# Liens utiles

- Comprendre les [Self-Organising Maps](https://en.wikipedia.org/wiki/Self-organizing_map) (SOM)

- Article scientifique sur les [Dynamic graph neural networks](https://www.researchgate.net/publication/2523357_A_Dynamic_Neural_Network_for_Continual) sur lequel se base ce projet.

- Article sur les [Self-Growing Neural Network](https://www.researchgate.net/publication/268454314_Anomaly_detection_using_dynamic_Neural_Networks_classification_of_prestack_data) (SGNN)

- Article sur le [réseau de neurones dynamique DAN2](https://www.researchgate.net/publication/256662765_A_Review_of_DAN2_Dynamic_Architecture_for_Artificial_Neural_Networks_Model_in_Time_Series_Forecasting)

- [Transformée de Fourier](https://helios2.mi.parisdescartes.fr/~eprovenz/include/Poly.pdf) et applications

- [Transformée en Ondelettes continues](https://www.weisang.com/fr/dokumentation/timefreqspectrumalgorithmscwt_fr/)

<br/>

# Librairies

Libraries utilisées :

<p align="center">
<a href="https://plotly.com/python/"><img src="https://img.shields.io/badge/Lib-Plotly-937BCB" alt="Plotly"></a>
<a href="https://github.com/talboger/fastdist"><img src="https://img.shields.io/badge/Lib-Fastdist-937BCB" alt="Fastdist"></a>
<a href="https://scipy.github.io/devdocs/index.html"><img src="https://img.shields.io/badge/Lib-Scipy-937BCB" alt="Scipy"></a>
<a href="https://librosa.org"><img src="https://img.shields.io/badge/Lib-Librosa-937BCB" alt="Librosa"></a>
<a href="https://pypi.org/project/fastdtw/"><img src="https://img.shields.io/badge/Lib-fastdtw-937BCB" alt="fastdtw"></a>
<a href="https://pypi.org/project/pickle5/"><img src="https://img.shields.io/badge/Lib-pickle-937BCB" alt="pickle"></a>
</p>
<br>

<p align="center">
  <a href="https://github.com/antonin-lfv" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97302854-e484da80-1859-11eb-9374-5b319ca51197.png" title="GitHub" width="40" height="40"></a>
  <a href="https://www.linkedin.com/in/antonin-lefevre-565b8b141" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303444-b2c04380-185a-11eb-8cfc-864c33a64e4b.png" title="LinkedIn" width="40" height="40"></a>
  <a href="mailto:antoninlefevre45@icloud.com" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303543-cec3e500-185a-11eb-8adc-c1364e2054a9.png" title="Mail" width="40" height="40"></a>
</p>


---------------------------

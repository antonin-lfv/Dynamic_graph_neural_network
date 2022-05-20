<p align="center">
	<img src="https://user-images.githubusercontent.com/63207451/114284722-45901b80-9a52-11eb-8a0c-e99fc8681436.gif" height="80" alt="">
	</p>

<h1 align="center">Dynamic graph neural network</h1>

<br>

<p align="center">
<a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.10-2ea44f" alt="Python - 3.10"></a>
<a href="https://en.wikipedia.org/wiki/Types_of_artificial_neural_networks#Dynamic"><img src="https://img.shields.io/badge/Dynamic-Neural_Network-018291" alt="Dynamic - Neural Network"></a>
<a href="https://github.com/antonin-lfv/Dynamic_graph_neural_network"><img src="https://img.shields.io/github/stars/antonin-lfv/Dynamic_graph_neural_network?style=social" alt="stars"></a>

</p>

<br>

<p align="center">
Les réseaux de neurones dynamiques sont une branche peu développée du Deep Learning, qui repose sur un principe simple, l'architecture du réseau est dynamique. Cela a plusieurs avantages, notamment le fait que le réseau est en perpetuel apprentissage, et qu'il peut changer sa structure et le routage des informations en fonction des données, ce qui le rend très flexible. Le dynamisme s'opère au niveau de la profondeur du réseau mais aussi sur sa largeur (dans le cas d'un réseau en couches comme dans le modèle DAN2). Ici, on va encore un peu plus loin dans ce concept de réseau de neurones dynamiques, car on va utiliser une structure de graphe, c'est à dire sans organisation en couches comme les modèles classiques. <br>
	</p>

<p align="center">
Concernant ce projet, il a pour objectif de tester le pouvoir classificateur d'un réseau de neurones dynamiques en graphe décrit par l'article scientifique dont le lien est ci-dessous (le deuxième). L'article n'évoque qu'une partie mathématique et quelques voix pour la mise en place du modèle. Ainsi, La première étape de ce projet sera d'implémenter la structure du graphe ainsi que les méthodes associées telles qu'elles sont décrites dans cet article, puis, en fonction des resultats, d'améliorer le modèle. On verra que l'utilisation de la transformée de Fourier sera nécessaire dans un premier temps. Ensuite, une partie graphique sera implémentée avec la librairie Plotly qui servira à s'assurer de la bonne mise en place des premières méthodes (uniquement sur la première partie de l'implémentation). Concernant les phases d'expérimentation, on tentera pour commencer de classer plusieurs types de fonctions classiques, puis nous poursuivrons sur une classification de signaux sinusoïdaux. Dans un premier temps, les signeaux sont tous de la même taille, puis nous élargirons le domaine de compétences du réseau à des signaux de tailles différentes. Dans ce repository, une première partie sera consacrée à l'aspect mathématique du modèle, pour mieux comprendre son fonctionnement. Puis sera expliqué l'implémentation avec Python avec les différents tests et résultats.
	</p>

<br>

<br>

> To Do :
> - optimiser le calcul du **Dynamic Time Warping**


<br>

# Liens utiles

- Comprendre les [Self-Organising Maps](https://en.wikipedia.org/wiki/Self-organizing_map) (SOM)

- Article scientifique sur les [Dynamic graph neural networks](https://www.researchgate.net/publication/2523357_A_Dynamic_Neural_Network_for_Continual) sur lequel se base ce projet.

- Article sur les [Self-Growing Neural Network](https://www.researchgate.net/publication/268454314_Anomaly_detection_using_dynamic_Neural_Networks_classification_of_prestack_data) (SGNN)

- Article sur le [réseau de neurones dynamique DAN2](https://www.researchgate.net/publication/256662765_A_Review_of_DAN2_Dynamic_Architecture_for_Artificial_Neural_Networks_Model_in_Time_Series_Forecasting)

- [Transformée de Fourier](https://helios2.mi.parisdescartes.fr/~eprovenz/include/Poly.pdf) et applications

<br/>

# Index

1. [Librairies](#librairies)
2. [Modèle mathématique](#modèle-mathématique)
    1. [Principe](#principe)
    2. [Prédictions](#prédictions)
3. [Implémentation](#implémentation)
4. [Expérimentations](#Expérimentations)
	1. [Classification de fonctions classiques](#1-Classification-de-fonctions-classiques)
	2. [Classification de signaux sinusoïdaux](#2-Classification-de-signaux-sinusoïdaux)
	3. [Classification de signaux soumis à une transformée de Fourier](#3-Classification-de-signaux-soumis-à-une-transformée-de-Fourier)
	4. [Classification de signaux avec la méthode Dynamic Time Warping](#4-Classification-de-signaux-avec-la-méthode-Dynamic-Time-Warping)
4. [Bonus](#bonus)
5. [Conclusion](#conclusion)

<br>

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

# Modèle mathématique

## Principe

Voici un réseau basique, chaque neurone $$x$$ contient un vecteur qui est de la même taille que l'input. Les liaisons entre les neurones sont des scalaires, et on note $$c_{ij}$$
la liaison entre le neurone $$i$$ et $$j$$. Les neurones les plus semblables sont connectés par un poids synaptique.

<p align="center">
	<img src="https://user-images.githubusercontent.com/63207451/165794912-0e449845-0544-4234-842b-fdd41a7c3e13.png" alt="archi of PSOM">
	</p>

A partir de là, le vecteur d'entrée noté ![formula](https://render.githubusercontent.com/render/math?math=u) est comparé avec le vecteur de chaque neurone. Le neurone le plus proche (avec une distance euclidienne notée ![formula](https://render.githubusercontent.com/render/math?math=d) ) de l'input est alors appelé le foyer, et est noté ![formula](https://render.githubusercontent.com/render/math?math=z(x) ).

Ainsi, soit ![formula](https://render.githubusercontent.com/render/math?math=x=[x_1,x_2,...,x_n]^T) un vecteur d'un neurone, et ![formula](https://render.githubusercontent.com/render/math?math=u=[u_1,u_2,...,u_n]^T)
le vecteur d'entrée, alors la distance euclidienne 
![formula](https://render.githubusercontent.com/render/math?math=d) entre ![formula](https://render.githubusercontent.com/render/math?math=x) et ![formula](https://render.githubusercontent.com/render/math?math=u) est définie par : <br>

<p align="center">
   <img src="https://render.githubusercontent.com/render/math?math=||d||_2=\left[\sum_{i=1}^m(x_i-u_i)^2\right]^{1/2}" alt="" width="200">
</p>

Donc 

<p align="center">
   <img src="https://render.githubusercontent.com/render/math?math=z(x)=arg\min_j||d||_2" alt="" width="150">
</p>

Avec ![formula](https://render.githubusercontent.com/render/math?math=j=1,2,3,...,l)  et ![formula](https://render.githubusercontent.com/render/math?math=l) le nombre de neurones dans le graphe.

Le neurone d'entrée est alors connecté aux neurones dont la similarité dépasse un certain seuil.

La distance euclidienne entre le vecteur d'entrée ![formula](https://render.githubusercontent.com/render/math?math=u) et le foyer ![formula](https://render.githubusercontent.com/render/math?math=z) est ensuite utilisé pour modifier le foyer (son vecteur). On introduit le scalaire ![formula](https://render.githubusercontent.com/render/math?math=b_v) un paramètre d'échelle qui correspond au learning rate du réseau.

<p align="center">
   <img src="https://render.githubusercontent.com/render/math?math=\Delta z(x)=b_v(z-u)" alt="" width="150">
</p>

Après modification du foyer, on va modifier de la même manière les neurones connectés à proximité du foyer (en dessous d'un certain seuil ![formula](https://render.githubusercontent.com/render/math?math=a_n) de similarité), mais à un degré moindre par rapport au foyer. On introduit le scalaire ![formula](https://render.githubusercontent.com/render/math?math=b_c) un paramètre d'échelle qui correspond au taux de changement du noeud. (![formula](https://render.githubusercontent.com/render/math?math=k) est le foyer) 

<p align="center">
   <img src="https://render.githubusercontent.com/render/math?math=\Delta x_j=b_c*c_{jk}(x_k-x_j)" alt="" width="200">
</p>

Avec ![formula](https://render.githubusercontent.com/render/math?math=k=1,2,3,..,l) ,  ![formula](https://render.githubusercontent.com/render/math?math=j!=k) ,  ![formula](https://render.githubusercontent.com/render/math?math=b_c) ∈ ![formula](https://render.githubusercontent.com/render/math?math=R) 

On réduit aussi les connexions du foyers, ce qui rapproche les neurones similaires. La force avec laquelle elles sont actualisées est le scalaire ![formula](https://render.githubusercontent.com/render/math?math=b_l) . La nouvelle valeur de la connexion entre ![formula](https://render.githubusercontent.com/render/math?math=j) et ![formula](https://render.githubusercontent.com/render/math?math=k) est alors ![formula](https://render.githubusercontent.com/render/math?math=c_{jk}=b_l(||x_j-x_k||))
Avec ![formula](https://render.githubusercontent.com/render/math?math=b_l) ∈ ![formula](https://render.githubusercontent.com/render/math?math=R)


Si l'entrée du réseau ![formula](https://render.githubusercontent.com/render/math?math=u) est complètement différente des autres neurones (en terme de distance euclidienne) alors un nouveau neurone ou groupe de neurones est ajouté et connecté au foyer. Un neurone est ajouté quand ![formula](https://render.githubusercontent.com/render/math?math=||d||>a_n)   avec ![formula](https://render.githubusercontent.com/render/math?math=a_n) ∈ ![formula](https://render.githubusercontent.com/render/math?math=R)
C'est à dire si la distance minimale entre l'entrée et les neurones dépasse le seuil ![formula](https://render.githubusercontent.com/render/math?math=a_n).


Élagage du réseau : On supprime les liens qui deviennent trop longs, c'est à dire soit ![formula](https://render.githubusercontent.com/render/math?math=a_r) le seuil, le lien entre le neurone ![formula](https://render.githubusercontent.com/render/math?math=i) et ![formula](https://render.githubusercontent.com/render/math?math=j) est supprimé si ![formula](https://render.githubusercontent.com/render/math?math=c_{ij}>a_r) . 

Quand un neurone n'a plus de lien, il est supprimé. (on préfèrera ici conserver tous les neurones)

## Prédictions


Il y a deux méthodes pour effectuer une prédiction

1) Par concaténation 

En considérant que chaque neurone contient un vecteur qui contient en plus la sortie souhaitée. Ainsi, en envoyant en entrée un vecteur avec seulement une taille correspondant à la taille de ceux des neurones sans la sortie, le réseau devrait renvoyer la sortie du foyer associé. 

> Cette méthode induit que les données sont labellisés 

2) Par labellisation automatique des clusters

Si la connexion entre deux neurones est suffisamment petite, le réseau va associer aux deux neurones le même label. Par exemple, sur l'image ci-dessous le neurone du Label A et le neurone du label B ont une liaison de poids 5, ce qui veut dire qu'ils sont très similaires, dans ce cas les labels sont fusionnés.

<p align="center">
	<img src="https://user-images.githubusercontent.com/63207451/165794988-62cc255f-0d0a-428d-ae8c-807dbd928c0c.png" alt="archi of PSOM">
	</p>

> Marche très bien pour des données non labélisées

<br>

On utilisera cette dernière méthode de prédiction dans ce projet.

<br>

# Implémentation

L'implémentation repose sur la création de deux classes. Une classe représentant les neurones (`Neuron`), et une classe représentant le graphe (`Graph`). Ainsi, chaque instance de graphe possède un certain nombre de neurones. <br>

La classe `Neuron` possède plusieurs paramètres : 
- vecteur : qui représente le vecteur du neurone, c'est sur ce vecteur que repose le modèle
- index : identifiant unique d'un neurone dans un graphe, il est attribué grâce à un compteur interne au graphe
- label : c'est la classe à laquelle le neurone appartient, il est attribué lors de l'ajout des neurones
- liaisons : qui est un dictionnaire des liaisons dont les clés représentent l'index d'un neurone, et la valeur son poids

La classe `Graph` possède également plusieurs paramètres :
- neurons : qui est un dictionnaire contenant tous les neurones du graphe, indexé par l'index des neurones
- compt_neurons : qui est initialisé à 0 lors de la création du graphe et qui correspond au compteur de neurones, pour l'attribution des index

Ces paramètres seront fixes tout au long de ce projet. Concernant les méthodes de ces deux classes, elles seront détaillées par la suite.

<br>

La première étape de la modélisation est la création du graphe et l'ajout de neurones. On définit alors la méthode `addNeuron` de la classe `Graph` prenant en paramètre un objet de la classe `Neuron`. <br>

On définit dans cette méthode 3 cas :
- Si le graphe est vide : le neurone prend comme label son index, et aucune liaison n'est alors créée.
- Si le graphe contient un seul neurone : on assigne au nouveau neurone le label du premier si la distance entre les deux est inférieure au seuil ![formula](https://render.githubusercontent.com/render/math?math=a_{n}), sinon son label est défini par son index. On crée ensuite la liaison entre les deux. (qui est ajouté aux deux neurones)
- Si il y a plus que deux neurones, on calcul le foyer du nouveau neurone. Si la distance entre les deux est inférieure au seuil ![formula](https://render.githubusercontent.com/render/math?math=a_{n}) il prend le label du foyer, et on connecte au nouveau neurone tous les autres à une distance inférieure à ![formula](https://render.githubusercontent.com/render/math?math=a_{n}). Sinon, l'index du nouveau neurone devient aussi son label, et il n'est connecté qu'a son foyer.

<br>

Dans le modèle initial proposé par l'article, après chaque ajout d'un neurone on doit, si le neurone tout juste ajouté est à une distance inférieure à ![formula](https://render.githubusercontent.com/render/math?math=a_{n}) de son foyer, modifier le foyer ainsi que toutes ces liaisons et neurones voisins. Si une liaison devient supérieure à ![formula](https://render.githubusercontent.com/render/math?math=a_{r}) durant cette modification alors la liaison est supprimée. (tous les voisins du foyer sont déjà par définition à une distance inférieure à ![formula](https://render.githubusercontent.com/render/math?math=a_{n})) 

<br>

On définit alors trois méthodes dans la classe `Neuron` qui vont permettre ces modifications :
- `alterFoyer` : qui va altérer le vecteur du foyer du nouveau neurone ajouté
- `alterVoisins` : qui va modifier les voisins du foyer du nouveau neurone ajouté, selon le modèle mathématique
- `alterLiaisons` : qui va altérer les liaisons du foyer du nouveau neurone ajouté selon le modèle mathématique, et supprimer celles qui deviennent supérieures à ![formula](https://render.githubusercontent.com/render/math?math=a_{r})

<br> 
Si un neurone n'a plus de connexion on lui attribut son label comme classe. <br>

<br>

Dans l'implémentation on ajoute une méthode `fit()` qui prend en paramètre un dictionnaire indexé sur les entiers naturels, contenant le tableau de valeurs de chaque neurone. Elle ajoute automatiquement les neurones avec la méthode `add_Neuron`.

<br>

Un problème dans l'implémentation de la fonction d'affichage du graphe apparaît, en effet, l'ajout d'un neurone assez proche de son foyer (distance inférieure à ![formula](https://render.githubusercontent.com/render/math?math=a_{n})) induit une modification du foyer et des voisins et liaisons de ce dernier. Ceci déséquilibre le lien mathématique (de distance euclidienne) entre les neurones et de ce fait, la méthode permettant d'afficher le graphe ne permettra pas de le faire. On se basera ainsi sur l'affichage des neurones (avec la méthode `__repr__` de chaque classe) du graphe avec leur label pour savoir comment le modèle les a rassemblés. 

<br>

# Expérimentations

<br>

## 1. Classification de fonctions classiques

On va dans cette première partie utiliser le modèle de la façon la plus basique possible. La distance euclidienne utilisée par le réseau nous oblige à avoir des fonctions avec le même nombre de points. Les signaux que nous comparerons seront uniquement soumis à la distance euclidienne. Prenons un ensemble de 10 neurones, dont les index **0, 3, 4, 6, 7** sont ceux représentants des fonctions cosinus (en bas) et **1, 2, 5, 8, 9** des fonctions racines (en haut). On peut les représenter graphiquement : <br>

<p align="center">
	<img width="950" alt="Capture d’écran 2022-05-08 à 11 35 48" src="https://user-images.githubusercontent.com/63207451/167290435-eb73a979-1e67-4d85-9172-935158159ec6.png">
	</p>

<br>

On obtient après ajout de ces neurones le graphe suivant : (on affiche la liste des neurones du graphe, les vecteurs ne sont pas affichés pour des raisons de lisibilité)

<br>

```
{
0: Neuron(index=0, liaisons={1: 15.04025}, label=0),
 1: Neuron(index=1, liaisons={0: 15.04025, 2: 0.9881875, 5: 6.126875, 8: 2.33225, 9: 18.5865}, label=1),
 2: Neuron(index=2, liaisons={1: 0.9881875, 3: 58.147}, label=1),
 3: Neuron(index=3, liaisons={2: 58.147, 4: 3.4405, 6: 2.06425, 7: 22.845}, label=3),
 4: Neuron(index=4, liaisons={3: 3.4405, 6: 6.976, 7: 24.364}, label=3),
 5: Neuron(index=5, liaisons={1: 6.126875, 6: 17.707, 7: 47.834}, label=1),
 6: Neuron(index=6, liaisons={3: 2.06425, 4: 6.976, 5: 17.707, 7: 10.631}, label=3),
 7: Neuron(index=7, liaisons={3: 22.845, 4: 24.364, 5: 47.834, 6: 10.631}, label=3),
 8: Neuron(index=8, liaisons={1: 2.33225, 9: 37.861}, label=1),
 9: Neuron(index=9, liaisons={1: 18.5865, 8: 37.861}, label=1)
 }
```

<br>

On remarque que les neurones d'index 1, 2, 5, 8 et 9 sont ajoutés à la même classe. Ce groupe de 5 neurones correspond exactement au type de fonction racine, donc le réseau a parfaitement réussi à regrouper ces données ensemble. De même que les neurones 3, 4, 6, 7 qui sont les neurones de type cosinus.
Mais dans cette dernière classe le neurone 0 a été classé en tant qu'outlier. (ce n'est pas totalement absurde dans le sens ou il sa fonction est éloignée de toutes les autres) <br>

On va poursuivre les tests avec d'autres données pour voir comment le modèle se comporte.

<br>

## 2. Classification de signaux sinusoïdaux

Dans cette deuxième partie, on va continuer à utiliser uniquement la distance euclidienne tel que décrit dans l'article. On prend alors 9 neurones, qui représentent des signaux quelconques qui sont des sommes aléatoires de fonctions sinusoïdales. On va alors tester différents seuils pour voir si on arrive à trouver une classification satisfaisante. <br>
On peut déjà tracer les courbes représentant les 9 neurones : <br>

<p align="center">
<img width="950" alt="Capture d’écran 2022-05-10 à 13 52 16" src="https://user-images.githubusercontent.com/63207451/167622241-c2550e20-d4d5-4ea3-9c60-152efcaf5319.png">
	</p>

On remarque des signaux de différentes périodicités, avec des amplitudes plus ou moins grandes. On va maintenant ajouter nos neurones au réseau pour voir comment le modèle va les rassembler. Voici le résultat : <br>

```
{
0: Neuron(index=0, vecteur="", liaisons={1: 21.28756, 2: 33.84723999999999, 4: 6.818349999999999, 5: 23.8938}, label=0),
 1: Neuron(index=1, vecteur="", liaisons={0: 21.28756, 3: 67.593}, label=1),
 2: Neuron(index=2, vecteur="", liaisons={0: 33.84723999999999}, label=2),
 3: Neuron(index=3, vecteur="", liaisons={1: 67.593, 6: 77.476}, label=3),
 4: Neuron(index=4, vecteur="", liaisons={0: 6.818349999999999}, label=0),
 5: Neuron(index=5, vecteur="", liaisons={0: 23.8938}, label=0),
 6: Neuron(index=6, vecteur="", liaisons={3: 77.476, 7: 35.44029999999999}, label=6),
 7: Neuron(index=7, vecteur="", liaisons={6: 35.44029999999999, 8: 17.2781}, label=7),
 8: Neuron(index=8, vecteur="", liaisons={7: 17.2781}, label=7)
 }
```

<br>

On remarque que certains neurones on été ajoutés au même ensemble. Les neurones 0, 4 et 5 appartiennent au même cluster, de même que les neurones 7 et 8. Enfin, les autres neurones sont classés dans des clusters différents.
On peut relever que le modèle a rassemblé les signaux qui se superposent bien, cependant il ne prend pas en compte le fait que les signaux sont périodiques et que deux signaux peuvent se superposer à une translation près. On peut alors comprendre pourquoi le réseau a réussi à classer de façon correcte les fonctions racines et sinus, c'est parce qu'elles sont superposables. <br>

<br>

On pourrait alors essayer de représenter ces signaux d'une autre manière, qui pourrait faire ressortir les caractéristiques de ces derniers pour mieux les comparer .. avec **une décomposition en séries de Fourier**.

<br>

## 3. Classification de signaux soumis à une transformée de Fourier

On va dans cette section utiliser la transformée de Fourier pour voir si le modèle réussi à mieux classer les signaux. On va prendre comme précédemment des signaux sinusoïdaux aléatoires. Le principe est donc le suivant : on va effectuer une transformée de Fourier sur chacun des signaux brutes, et le résultat de chacune des transformations est alors passé aux neurones. Cette manipulation va permettre au réseau de ne pas être trompé entre deux signaux en moyenne identiques, et parfaitement superposables. C'est donc entre les transformées de Fourier des signaux que le modèle va faire les calculs de distance euclidienne. <br>

Voici les 16 signaux, correspondant aux 16 neurones du réseau : <br>

<p align="center">
<img width="950" alt="Capture d’écran 2022-05-10 à 20 00 48" src="https://user-images.githubusercontent.com/63207451/167693006-dcd7b083-0ded-4876-85cf-aba57e3fc1bd.png">
	</p>

<br>

On observe à vue d'oeil des différences au niveau des fréquences. Appliquons maintenant une transformée de Fourier sur chacun d'entre eux. Voici les résultats : <br>

<p align="center">
<img width="950" alt="Capture d’écran 2022-05-10 à 20 03 00" src="https://user-images.githubusercontent.com/63207451/167693362-e56e97c7-d8a3-484b-857e-00f2d3e4d8ac.png">
	</p>

<br>

Ce sont ces signaux qui seront passés aux neurones. On rappelle que soit ![formula](https://render.githubusercontent.com/render/math?math=f) notre signal, alors sa transformée de Fourier et la fonction ![formula](https://render.githubusercontent.com/render/math?math=F(f)) définie par : 

<p align="center">
   <img src="https://render.githubusercontent.com/render/math?math=F(f)=\int_{-\infty}^{+\infty} f(x)e^{-ix \xi} dx" alt="FFT" width="250">
	</p>
	
<br>

Si on regarde les données brutes du réseau on obtient ceci : <br>

```
{
0: Neuron(index=0, liaisons={1: 0.006926400000000001, 3: 0.164, 4: 1.0906, 7: 2.1456, 8: 1.1800000000000002, 11: 1.1814, 15: 3.769}, label=0),
 1: Neuron(index=1, liaisons={0: 0.006926400000000001, 2: 0.30952, 3: 0.028696000000000006, 4: 1.1492000000000002, 8: 1.2998, 11: 1.2300000000000002, 15: 0.5498000000000001}, label=0),
 2: Neuron(index=2, liaisons={1: 0.30952, 15: 6.163}, label=2),
 3: Neuron(index=3, liaisons={0: 0.164, 1: 0.028696000000000006, 4: 1.0162000000000002, 5: 0.32352000000000003}, label=0),
 4: Neuron(index=4, liaisons={0: 1.0906, 1: 1.1492000000000002, 3: 1.0162000000000002, 8: 5.77, 11: 5.981, 15: 3.842}, label=0),
 5: Neuron(index=5, liaisons={3: 0.32352000000000003, 6: 0.21260000000000004, 8: 0.9372, 11: 6.406}, label=5),
 6: Neuron(index=6, liaisons={5: 0.21260000000000004, 8: 4.875, 11: 6.383}, label=5),
 7: Neuron(index=7, liaisons={0: 2.1456, 9: 7.783, 10: 0.26996000000000003}, label=7),
 8: Neuron(index=8, liaisons={0: 1.1800000000000002, 1: 1.2998, 4: 5.77, 5: 0.9372, 6: 4.875, 11: 6.268, 15: 4.79}, label=5),
 9: Neuron(index=9, liaisons={7: 7.783}, label=9),
 10: Neuron(index=10, liaisons={7: 0.26996000000000003, 12: 0.03649600000000001, 13: 1.0204000000000002, 14: 1.0712}, label=10),
 11: Neuron(index=11, liaisons={0: 1.1814, 1: 1.2300000000000002, 4: 5.981, 5: 6.406, 6: 6.383, 8: 6.268, 15: 4.869}, label=0),
 12: Neuron(index=12, liaisons={10: 0.03649600000000001, 13: 0.979, 14: 5.743}, label=10),
 13: Neuron(index=13, liaisons={10: 1.0204000000000002, 12: 0.979, 14: 5.648}, label=10),
 14: Neuron(index=14, liaisons={10: 1.0712, 12: 5.743, 13: 5.648}, label=10),
 15: Neuron(index=15, liaisons={0: 3.769, 1: 0.5498000000000001, 2: 6.163, 4: 3.842, 8: 4.79, 11: 4.869}, label=0)
 }
 ```

<br>

De façon plus lisible, voici comment le réseau a classé les signaux par label (1 colonne = 1 cluster de neurones) : <br>

| Clusters | Signaux |
| --- | ----------- |
| <img width="300" alt="Capture d’écran 2022-05-10 à 20 42 22" src="https://user-images.githubusercontent.com/63207451/167699829-1e656a93-aa86-4b33-a6a2-b5b2b04404a5.png"> | <img width="950" alt="Capture d’écran 2022-05-11 à 11 34 49" src="https://user-images.githubusercontent.com/63207451/167818769-bab5ada1-e4b8-443c-9f69-886bf8915425.png"> |

<br>

On peut ainsi remarquer que cette fois ci, la classification est plutôt très bien réussie. Les neurones de label 0 sont les signaux avec la fréquence la plus basse, et les neurones de label 10 sont les signaux de plus hautes fréquences. Les labels intermédiaires sont des signaux de fréquences moyennes (par rapport aux labels 0 et 10), et donc la classification est un peu plus compliquée. <br>

On peut réitérer la classification avec d'autres données, voici certains résultats :

<br>

| Clusters | Signaux |
| --- | ----------- |
| <img width="300" alt="Capture d’écran 2022-05-11 à 11 44 01" src="https://user-images.githubusercontent.com/63207451/167820535-573c95fc-d92e-4607-9364-2b267acd97d6.png"> |<img width="950" alt="Capture d’écran 2022-05-11 à 11 39 13" src="https://user-images.githubusercontent.com/63207451/167819597-9a7c77ce-052b-4dab-bba0-b7a36913a7d4.png"> |

<br>

| Clusters | Signaux |
| --- | ----------- |
| <img width="300" alt="Capture d’écran 2022-05-11 à 11 44 09" src="https://user-images.githubusercontent.com/63207451/167820573-d5f9a896-e60b-497c-b66f-9eaef122f3b7.png"> | <img width="950" alt="Capture d’écran 2022-05-11 à 11 42 14" src="https://user-images.githubusercontent.com/63207451/167820128-908192cc-3327-48dd-b21e-0eb741b3271e.png"> |

<br>

Il n'y a pas d'erreurs aberrantes, contrairement aux configurations précédentes, et ce sur une multitude de données. On peut alors valider cette technique (d'application du FFT) comme étant efficace, et aussi valider le modèle décrit dans l'article sur lequel se base ce projet. (tout en ajoutant cette petite subtilité dans le passage des signaux aux neurones) <br>

<br>

Cependant, ce modèle a une limite dans son implémentation actuelle (telle que décrite dans l'article). En effet, tout est basé sur la distance euclidienne entre les neurones, ce qui immplique que les signaux doivent impérativement être de la même taille. Ainsi, il faudrait utiliser une méthode pour calculer la distance entre deux vecteurs de tailles différentes ... la méthode de **Dynamic Time Warping**.

<br>

## 4. Classification de signaux avec la méthode Dynamic Time Warping

<br>

Dans cette section, nous allons changer la façon de calculer les distances à l'interieur du réseau. En effet, la distance euclidienne ne convient que pour des signaux de même taille. Si on calcule la distance euclidienne entre un vecteur de taille `n` et un autre de taille `m` tel que `n<m` alors cela revient à calculer la distance entre deux vecteur de taille `n`. (le vecteur de taille `m` est tronqué) <br>

<br>

Ainsi, on se passera de la Transformée de Fourier dont le but était justement de pouvoir comparer des signaux qui était superposables à une translation près, et on va simplememtn changer la fonction qui calcule la distance entre les vecteurs des neurones, en utilisant la méthode Dynamic Time Warping (DTW) 

<br>

<br>

On peut représenter graphiquement la distance euclidienne comme ceci : <br>

<br>

<p align="center">
<img width="500" alt="Capture d’écran 2022-05-17 à 11 13 46" src="https://user-images.githubusercontent.com/63207451/168775722-ef3ad9f7-0ddc-4a2a-b92b-48e0cdd1315d.png">
	</p>

Le signal rouge est le vecteur de taille `n`, il est plus petit que le signal bleu de taille `m`. On peut alors remarqué que la distance euclidienne entre les deux sera grande, car les deux signaux, malgrès leur ressemblance, ne sont pas alignés. Et la distance euclidienne compare les valeurs une à une dans l'ordre. Ils sont alignés à une translation près, comme ce qu'on avait remarqué dans la partie 2. <br>

<br>

Pour palier à ce problème, il existe la méthode **Dynamic Time Warping**. Cette méthode permet de trouver l’alignement global optimal entre deux signaux, c’est-à-dire d’associer chaque élément de chaque signal à au moins un élément de l’autre signal en minimisant les coûts d’association. Le coût d’une association correspond à la distance entre les deux éléments. Le résultat numérique fournit par DTW correspond à la somme des hauteurs des “barreaux” formés par les associations (les barres noires entre les signaux rouge et bleu). On remarque sur la figure ci-dessous à gauche des signaux que DTW a réaligné correctement les deux signaux, et parvient ainsi à saisir des similarités que la distance euclidienne ne peut extraire.

<br>

<p align="center">
	<img width="500" alt="Capture d’écran 2022-05-17 à 11 18 56" src="https://user-images.githubusercontent.com/63207451/168776805-50682d7f-29c6-4eda-af09-5bb4466a1504.png">
	</p>

<br>

Cette fois-ci nous allons utiliser de vraies données, qui sont des chants d'oiseaux. On isolera les syllabes de plusieurs espèces, ce qui constituera nos données d'entrées. Puis, on passera au modèle ces données qui utilisera la méthode DTW pour calculer les distances. 

<br>

Voici des exemples de syllabes de chants d'oiseaux: 

<br>

<p align="center">
	<img width="500" alt="syll1" src="https://user-images.githubusercontent.com/63207451/168826924-ca1147e4-78be-495f-912d-dd408f406c26.png">
	<img width="500" alt="syll2" src="https://user-images.githubusercontent.com/63207451/168826940-cdd69f52-584e-44e7-bbd4-8ce3ddd9afd7.png">
	</p>

<br>

On remarque que les deux syllabes sont de tailles différentes, c'est ce qui motive l'utilisation du DTW. Maintenant, voici le dataset de syllabes de chants d'oiseaux que nous allons utiliser dans la suite des tests:

<br>

<p align="center">
<img width="950" alt="data" src="https://user-images.githubusercontent.com/63207451/169101252-e1916ba9-438d-4760-99cc-e524192e4805.png">
	</p>

<br>

Les neurones 0 à 4 contiennent des syllabes d'une espèce d'oiseau, et les neurones 5 à 9 contiennent des syllabes d'une autre espèce. On ajoute maintenant les neurones au réseau, et on observe la classification du modèle :

<br>

<p align="center">
<img width="950" alt="cluster" src="https://user-images.githubusercontent.com/63207451/169101745-19858034-7a21-4428-a1bc-87d20820ecbb.png">
	</p>

<br>

Tout d'abord, on s'aperçoit que toute la première espèce d'oiseau a été associée au même cluster (de label 0). Pour la deuxième espèce, le résultat est plus mitigé, en effet l'espèce a été divisée en trois sous-catégories (labels 5, 6 et 9). En fait, notre modèle a un pouvoir de classification trop élevé pour nos données. Il cherche a vraiment trouver les différences entre les signaux. Mais, la classification par le modèle reste stable, en effet après avoir essayé une multitude de seuils différents le réseau ne classe quasiment jamais deux signaux d'espèces différentes ensemble, ce qui est très encourageant en terme de véracité du modèle.

<br>

À noter que le calcul de la distance avec la méthode DTW est très long, et donc pour un nombre de neurones plus élevé, le temps de calcul sera très grand. En terme de rapidité c'est la méthode par transformée de Fourier qui l'emporte. On peut comparer le temps d'exécution des deux méthodes :

<br>

<p align="center">
	<img width="1050" alt="temps exec" src="https://user-images.githubusercontent.com/63207451/169149156-e9cc45c9-91fa-402c-8285-cbff1057fffa.png">
	</p>

<br>

Nous avons donc réussi à développer dans un premier temps un modèle qui classifie des signaux de même taille. Nous avons utilisé la transformée de Fourier pour permettre au réseau de ne pas se faire tromper sur des signaux ressemblant à une translation près. Puis, nous avons élargies ses compétences en lui permettant d'utiliser une autre méthode de calcul des distances, la méthode de DTW, qui permet de calculer la ressemblance entre deux signaux de tailles différentes. Les tests sont assez concluants, mais il reste une chose sur laquelle discuter, qui concerne les seuils. Le modèle comporte en effet 5 seuils. Avec la pratique on peut fixer très rapidement les scalaires **bv** **bc** et **bl** en leur donnant une valeur aux alentours de 0.30, cependant, les 2 autres seuils, eux, dépendent complétement des données que l'on donne au modèle (sauf dans le cas de l'utilisation de la transformée de Fourier où les 2 derniers seuils tournent autour de 10). Ainsi, lors l'utilisation de la méthode DTW, avant de lancer une classification sur des données inconnues, il faudrait "calibrer" ces 2 derniers seuils avec des données connues qui sont à la même échelle que les données que l'on passera au modèle.

<br>
	
# Bonus

Cette partie est un bonus qui a été developpé dans un premier temps pour afficher un réseau de neurones en graphe lorsque les distances entre neurones ne tiennent compte **uniquement** des distances euclidiennes entre ces derniers. Dans le modèle détaillé dans ce projet, les neurones sont modifiés, et donc on ne peut pas les représenter avec cette méthode. <br>


Les données que nous avons à disposition sont les neurones avec leur vecteur. Nous calculerons toutes les distances nécessaires. Le but est alors de générer les coordonnées des neurones pour pouvoir les afficher avec Plotly. On se ramène à un problème purement mathématique, **comment placer `n` points en ne connaissant que les distances entre eux**. Pour cela on va utiliser une méthode geométrique consistant à trouver l'intersection de n cercles, grâce à un système à n équations non linéaires.
Voici les étapes de l'algoritme : <br>

- Étape 1: On place le premier neurone à la position (x=0, y=0) <br>
- Étape 2: Le deuxième neurone est translaté sur l'axe des x par rapport au premier neurone, ainsi en notant `d` la distance entre les deux neurones, le deuxième neurone est alors en position (d, 0) <br>
- Étape 3: Pour les autres neurones, on résout un système de n équation à 2 inconnues qui permet de trouver le neurone à l'intersection des cercles dont tous les autres points en sont les centres. 

Prenons l'exemple ou nous avons les 2 premiers neurones A et B, et nous voulons ajouter un 3e neurone C : <br>

Le premier point A est en coordonnées (x1=0, y1=0) avec comme rayon r1=2 qui est la distance entre A et le neurone C à ajouter
Le deuxième point B est en coordonnées (x2=3, y2=0) avec comme rayon r2=4 qui est la distance entre B et C

Ainsi, mathématiquement pour trouver les coordonnées (x, y) des intersections entre les 2 cercles de centre A et B voici le système : <br>

<p align="center">
	<img src="https://user-images.githubusercontent.com/63207451/166120794-b67cd845-33bf-4aa5-9b2d-ef12b7968836.png" alt="eq_syst">
	</p>

Pour choisir quel point prendre, on laissera la fonction de scipy nommée fsolve choisir. <br>
On obtient mathématiquement :

<br>

| mathématiquement | géométriquement |
| --- | ----------- |
|<img width="350" alt="2_cercles_code" src="https://user-images.githubusercontent.com/63207451/166466584-59ad449e-5083-425e-a8af-e6044a929975.png">|<img width="900" alt="2_cercles_plot" src="https://user-images.githubusercontent.com/63207451/166466707-760ad15d-2e26-453b-a548-d0e655002be8.png" >|
	
<br>

Puis, pour chaque nouveau neurone à ajouter, on ajoute une équation au système, ce qui nous donne les coordonnées (x, y) du nouveau neurone. Voici les résultats sur les deux prochains points :

<br>

On cherche les coordonnées du point D :

| mathématiquement | géométriquement |
| --- | ----------- |
|<img width="350" alt="3_cercles_code" src="https://user-images.githubusercontent.com/63207451/166467072-d4efe888-1aac-4055-a3d0-f5958b186c0e.png">|<img width="900" alt="3_cercles_plot" src="https://user-images.githubusercontent.com/63207451/166467089-b45760ca-fc6e-4551-b3b5-b9492b349963.png">|

<br> 

On cherche les coordonnées du point E :

| mathématiquement | géométriquement |
| --- | ----------- |
|<img width="450" alt="4_cercles_code" src="https://user-images.githubusercontent.com/63207451/166467120-efb5f312-e0ad-4742-9735-94d5081cc9eb.png">|<img width="800" alt="4_cercles_plot" src="https://user-images.githubusercontent.com/63207451/166467154-d220c01d-54ca-4a49-838a-a9a16f8671da.png">|

<br>

Voici un exemple de résultat obtenu avec 10 neurones :

<p align="center">
<img width="950" alt="3_connexions" src="https://user-images.githubusercontent.com/63207451/167640201-b1e42cdd-89c1-471c-8254-734d1b1280e5.png">
	</p>
<br>

Les 5 neurones à gauche représentent des fonctions racines, et les 5 neurones de droites représentent des fonctions sinus. Cette représentation est seulement basée sur la distance euclidienne entre les neurones. Comme ces fonctions sont très distinctes, les neurones sont bien séparés. Mais comme vu dans ce projet, la distance des fonctions n'est pas suffisante pour avoir un modèle de classification performant.<br>

<br>

# Conclusion

<br>

Ce projet nous a ammené, au travers de cette branche des réseaux de neurones dynamiques, à découvrir de nouveaux concepts et de nouvelles représentations des données. On peut dire que ce type de réseau en graphe est très efficace pour classer des signaux notamment, et aussi pour détecter des outliers (des anomalies). Mais, comme les résultats l'ont montré, il ne faut pas seulement travailler avec les signaux brutes, mais plutôt avec leur transformée de Fourier (Ou autre méthode, qui pourrait faire l'objet d'une nouvelle partie dans ce projet) qui permet au réseau de généraliser les types de signaux de façon plus efficace et sans faire d'erreur grossière.

<br>

<p align="center"><a href="#librairies"><img src="http://randojs.com/images/backToTopButton.png" alt="Haut de la page" height="29"/></a></p>

<br>

<br>

<p align="center">
  <a href="https://github.com/antonin-lfv" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97302854-e484da80-1859-11eb-9374-5b319ca51197.png" title="GitHub" width="40" height="40"></a>
  <a href="https://www.linkedin.com/in/antonin-lefevre-565b8b141" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303444-b2c04380-185a-11eb-8cfc-864c33a64e4b.png" title="LinkedIn" width="40" height="40"></a>
  <a href="mailto:antoninlefevre45@icloud.com" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303543-cec3e500-185a-11eb-8adc-c1364e2054a9.png" title="Mail" width="40" height="40"></a>
</p>


---------------------------

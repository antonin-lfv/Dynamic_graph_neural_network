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
Ce projet a pour objectif de tester le pouvoir classificateur d'un Dynamic graph neural network aux travers de plusieurs tests. Le modèle sur lequel est basé cet approche est disponible dans les liens utiles en dessous. La première étape sera d'implémenter la structure du graphe ainsi que les méthodes associées telles qu'elles sont décrites dans l'article, puis, en fonction des resultats, d'améliorer ce modèle sur le plan de l'implémentation. Ensuite, une partie graphique sera implémentée avec la librairie Plotly qui servira à suivre l'évolution architecturale du réseau (uniquement sur la première partie de l'implémentation). Concernant les phases d'expérimentation, on tentera de classer plusieurs types de fonctions, et de jouer avec les seuils présents dans le modèle. Dans ce repo, une première partie sera consacrée à l'aspect mathématique du modèle, pour mieux comprendre son fonctionnement. Puis sera expliqué l'implémentation avec Python avec les différents tests et résultats.
</p>

<br>

# Liens utiles

- Comprendre les [Self-Organising Maps](https://en.wikipedia.org/wiki/Self-organizing_map) (SOM)

- Article scientifique sur les [Dynamic graph neural networks](https://www.researchgate.net/publication/2523357_A_Dynamic_Neural_Network_for_Continual) sur lequel se base ce projet.

- Article sur les [Self-Growing Neural Network](https://www.researchgate.net/publication/268454314_Anomaly_detection_using_dynamic_Neural_Networks_classification_of_prestack_data) (SGNN)

<br/>

# Index

1. [Librairies](#librairies)
2. [Modèle mathématique](#modèle-mathématique)
    1. [Principe](#principe)
    2. [Prédictions](#prédictions)
3. [Implémentation](#implémentation)
	1. [Test sur des fonctions classiques](#1-Premier-test-sur-des-fonctions-classiques)
	2. [Test sur des signaux sinusoïdaux](#2-Deuxième-test-avec-des-signaux-sinusoïdaux)
	3. [Test avec des signaux soumis à une transformée de Fourrier](#3-Troisième-test-avec-utilisation-de-la-transormée-de-Fourrier)
4. [Bonus](#bonus)
5. [Conclusion](#conclusion)

<br>

# Librairies

Libraries utilisées :

<p align="center">
<a href="https://plotly.com/python/"><img src="https://img.shields.io/badge/Lib-Plotly-937BCB" alt="Plotly"></a>
<a href="https://github.com/talboger/fastdist"><img src="https://img.shields.io/badge/Lib-Fastdist-937BCB" alt="Fastdist"></a>
<a href="https://scipy.github.io/devdocs/index.html"><img src="https://img.shields.io/badge/Lib-Scipy-937BCB" alt="Scipy"></a>
</p>

<br>

# Modèle mathématique

## Principe

Voici un réseau basique, chaque neurone ![formula](https://render.githubusercontent.com/render/math?math=x) contient un vecteur qui est de la même taille que l'input. Les liaisons entre les neurones sont des scalaires, et on note ![formula](https://render.githubusercontent.com/render/math?math=c_{ij})
la liaison entre le neurone ![formula](https://render.githubusercontent.com/render/math?math=i) et ![formula](https://render.githubusercontent.com/render/math?math=j). Les neurones les plus semblables sont connectés par un poids synaptique.

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

Un problème dans l'implémentation de la fonction d'affichage du graphe apparaît, en effet, l'ajout d'un neurone assez proche de son foyer (distance inférieure à ![formula](https://render.githubusercontent.com/render/math?math=a_{n})) induit une modification du foyer et des voisins et liaisons de ce dernier. Ceci déséquilibre le lien mathématique (de distance euclidienne) entre les neurones et de ce fait, la méthode permettant d'afficher le graphe ne permettra pas de le faire. On se basera ainsi sur l'affichage des neurones (avec la méthode `__repr__` de chaque classe) du graphe avec leur label pour savoir comment le modèle les a rassemblés. 

<br>

### 1. Premier test sur des fonctions classiques

Prenons un ensemble de 10 neurones, dont les index **0, 3, 4, 6, 7** sont ceux représentants des fonctions cosinus (en bas) et **1, 2, 5, 8, 9** des fonctions racines (en haut). On peut les représenter graphiquement : <br>

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

On va poursuivre les tests avec d'autres données, et un peu plus de types différents.

<br>

### 2. Deuxième test avec des signaux sinusoïdaux

On prend ici 9 neurones, qui représentent des signaux quelconques qui sont des sommes aléatoires de fonctions sinusoïdales. On va alors tester différents seuils pour voir si on arrive à trouver une classification satisfaisante. <br>
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

On remarque que certains neurones on été ajouté au même ensemble. Les neurones 0, 4 et 5 appartiennent au même cluster, de même que les neurones 7et 8. Enfin, les autres neurones sont classés dans des cluster différents.
On peut relever que le modèle a rassembler les signaux qui se superposent bien, cependant il ne prend pas en compte le fait que les signaux sont périodiques et que deux signaux peuvent se superposer à une translation près. <br>

<br>

On pourrait alors essayer de représenter ces signaux d'une autre manière, qui pourrait faire ressortir les caractéristiques de ces derniers pour mieux les comparer .. avec **une décomposition en séries de Fourier**.

<br>

### 3. Troisième test avec utilisation de la transormée de Fourrier

On va dans cette section utiliser la transformée de Fourrier pour voir si le modèle réussi mieux à classer les signaux. On va prendre comme précédemment des signaux sinusoïdaux. Le principe est donc le suivant : on va effectuer une transformée de Fourrier sur chacun des signaux brutes, et le résultat de chacune des transformations est alors passé aux neurones. <br>

Voici les 16 signaux, correspondant aux 16 neurones du réseau : <br>

<p align="center">
<img width="950" alt="Capture d’écran 2022-05-10 à 20 00 48" src="https://user-images.githubusercontent.com/63207451/167693006-dcd7b083-0ded-4876-85cf-aba57e3fc1bd.png">
	</p>

<br>

On observe à vue d'oeil des différences au niveau des fréquences. Appliquons maintenant une transformée de Fourrier sur chacun d'entre eux. Voici les résultat : <br>

<p align="center">
<img width="950" alt="Capture d’écran 2022-05-10 à 20 03 00" src="https://user-images.githubusercontent.com/63207451/167693362-e56e97c7-d8a3-484b-857e-00f2d3e4d8ac.png">
	</p>

<br>

Ce sont ces signaux qui seront passés aux neurones. On rappelle que soit ![formula](https://render.githubusercontent.com/render/math?math=f) notre signal, alors sa transformée de Fourier et la fonction ![formula](https://render.githubusercontent.com/render/math?math=F(f)) définie par : 

<p align="center">
   <img src="https://render.githubusercontent.com/render/math?math=\int_{-\infty}^{+\infty} f(x)e^{-ix \xi} dx" alt="FFT" width="200">
	</p>
	
<br>


<br>

# Bonus

Cette partie est un bonus qui a été developpé dans un premier temps pour afficher un réseau de neurones en graphe lorsque les distances entre neurones ne tiennent compte **uniquement** des distances euclidiennes entre ces derniers. Dans le modèle détaillé dans ce projet, les neurones sont modifiés, et donc on ne peut pas les représenter avec cette méthode. <br>


Les données que nous avons à disposition sont les neurones avec leurs vecteur ainsi que les liaisons entre eux. Nous calculerons toutes les distances nécessaires. Le but est alors de générer les coordonnées des neurones pour pouvoir les afficher avec Plotly. On se ramène à un problème purement mathématique, comment placer `n` points en ne connaissant que la distance entre eux. Pour cela on va utiliser une méthode geométrique consistant à trouver l'intersection de n cercles, grâce à un système à n équations non linéaires.
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

Pour choisir quel point prendre, on laissera la fonction de scipy nommé fsolve choisir. <br>
On obtient mathématiquement :

<br>

<p align="center">
<img width="355" alt="2_cercles_code" src="https://user-images.githubusercontent.com/63207451/166466584-59ad449e-5083-425e-a8af-e6044a929975.png">
	</p>
	
<br>

et géométriquement : <br>

<p align="center">
<img width="500" alt="2_cercles_plot" src="https://user-images.githubusercontent.com/63207451/166466707-760ad15d-2e26-453b-a548-d0e655002be8.png" >
	</p>
<br>

Puis, pour chaque nouveau neurone à ajouter, on ajoute une équation au système, ce qui nous donne les coordonnées (x, y) du nouveau neurone. <br>

Voici les essaies sur les deux prochains points :

<br>

On cherche les coordonnées du point D :

<p align="center">
<img width="521" alt="3_cercles_code" src="https://user-images.githubusercontent.com/63207451/166467072-d4efe888-1aac-4055-a3d0-f5958b186c0e.png">
	</p>

<br>

<p align="center">
<img width="500" alt="3_cercles_plot" src="https://user-images.githubusercontent.com/63207451/166467089-b45760ca-fc6e-4551-b3b5-b9492b349963.png">
	</p>

<br> 

On cherche les coordonnées du point E :

<p align="center">
<img width="640" alt="4_cercles_code" src="https://user-images.githubusercontent.com/63207451/166467120-efb5f312-e0ad-4742-9735-94d5081cc9eb.png">
	</p>

<br>

<p align="center">
<img width="500" alt="4_cercles_plot" src="https://user-images.githubusercontent.com/63207451/166467154-d220c01d-54ca-4a49-838a-a9a16f8671da.png">
	</p>

<br>

Voici un exemple de résultat obtenu avec 10 neurones :

<p align="center">
<img width="950" alt="3_connexions" src="https://user-images.githubusercontent.com/63207451/167640201-b1e42cdd-89c1-471c-8254-734d1b1280e5.png">
	</p>
<br>

Les 5 neurones à gauche représentent des fonctions racines, et les 5 neurones de droites représentent des fonctions sinus. Cette représentation est seulement basée sur la distance euclidienne entre les neurones. <br>

<br>

# Conclusion

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

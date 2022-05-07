<p align="center">
	<img src="https://user-images.githubusercontent.com/63207451/114284722-45901b80-9a52-11eb-8a0c-e99fc8681436.gif" height="80" alt="">
	</p>

<h1 align="center">Dynamic graph neural network</h1>

<br/>

<p align="center">
<a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.10-2ea44f" alt="Python - 3.10"></a>
<a href="https://en.wikipedia.org/wiki/Types_of_artificial_neural_networks#Dynamic"><img src="https://img.shields.io/badge/Dynamic-Neural_Network-018291" alt="Dynamic - Neural Network"></a>
<a href="https://github.com/antonin-lfv/Dynamic_graph_neural_network"><img src="https://img.shields.io/github/stars/antonin-lfv/Dynamic_graph_neural_network?style=social" alt="stars"></a>

</p>

<br>

<p align="center">
Ce projet a pour objectif de tester le pouvoir classificateur d'un Dynamic graph neural network aux travers de plusieurs tests. Le modèle sur lequel est basé cet approche est disponible dans les liens utiles en dessous. La première étape sera d'implémenter la structure du graphe ainsi que les méthodes associées telles qu'elles sont décrites dans l'article, puis, en fonction des resultats, de modifier ce modèle pour en proposer un nouveau. Ensuite, une partie graphique sera implémentée avec la librairie Plotly qui servira à suivre l'évolution architecturale du réseau. Concernant les phases d'expérimentation, on tentera de classer plusieurs types de fonctions, et de jouer avec les seuils présents dans le modèle. Dans ce repo, une première partie sera consacrée à l'aspect mathématique du modèle, pour mieux comprendre son fonctionnement. Puis sera expliqué l'implémentation avec Python avec les différents tests et résultats.
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
   1. [Ajout des neurones](#ajout-des-neurones)
   2. [Apprentissage et prédiction](#apprentissage-et-prédiction)
      1. [Version de l'article](#version-de-larticle)
      2. [Version modifiéé](#version-modifiée)
   3. [Affichage du graphe](#affichage-du-graphe)

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
   <img src="https://render.githubusercontent.com/render/math?math=||d||_2=\left[\sum_{i=1}^m(x_i-u_i)^2\right]^{1/2}" alt="">
</p>

Donc 

<p align="center">
   <img src="https://render.githubusercontent.com/render/math?math=z(x)=arg\min_j||d||_2" alt="">
</p>

Avec ![formula](https://render.githubusercontent.com/render/math?math=j=1,2,3,...,l)  et ![formula](https://render.githubusercontent.com/render/math?math=l) le nombre de neurones dans le graphe.

Le neurone d'entrée est alors connecté aux neurones dont la similarité dépasse un certain seuil. (le même seuil qui, à partir duquel, un lien est supprimé)

La distance euclidienne entre le vecteur d'entrée ![formula](https://render.githubusercontent.com/render/math?math=u) et le foyer ![formula](https://render.githubusercontent.com/render/math?math=z) est ensuite utilisé pour modifier le foyer (son vecteur). On introduit le scalaire ![formula](https://render.githubusercontent.com/render/math?math=b_v) un paramètre d'échelle qui correspond au learning rate du réseau.

<p align="center">
   <img src="https://render.githubusercontent.com/render/math?math=\Delta z(x)=b_v(z-u)" alt="">
</p>

Après modification du foyer, on va modifier de la même manière les neurones connectés à proximité du foyer (en dessous d'un certain seuil ![formula](https://render.githubusercontent.com/render/math?math=a_n) de similarité), mais à un degré moindre par rapport au foyer. On introduit le scalaire ![formula](https://render.githubusercontent.com/render/math?math=b_c) un paramètre d'échelle qui correspond au taux de changement du noeud. (![formula](https://render.githubusercontent.com/render/math?math=k) est le foyer) 

<p align="center">
   <img src="https://render.githubusercontent.com/render/math?math=\Delta x_j=b_c*c_{jk}(x_k-x_j)" alt="">
</p>

Avec ![formula](https://render.githubusercontent.com/render/math?math=k=1,2,3,..,l) ,  ![formula](https://render.githubusercontent.com/render/math?math=j!=k) ,  ![formula](https://render.githubusercontent.com/render/math?math=b_c) ∈ ![formula](https://render.githubusercontent.com/render/math?math=R) 

On réduit aussi les connexions du foyers, ce qui rapproche les neurones similaires. La force avec laquelle elles sont actualisées est le scalaire ![formula](https://render.githubusercontent.com/render/math?math=b_l) . La nouvelle valeur de la connexion entre ![formula](https://render.githubusercontent.com/render/math?math=j) et ![formula](https://render.githubusercontent.com/render/math?math=k) est alors ![formula](https://render.githubusercontent.com/render/math?math=c_{jk}=b_l(||x_j-x_k||))
Avec ![formula](https://render.githubusercontent.com/render/math?math=b_l) ∈ ![formula](https://render.githubusercontent.com/render/math?math=R)


Si l'entrée du réseau ![formula](https://render.githubusercontent.com/render/math?math=u) est complètement différente des autres neurones (en terme de distance euclidienne) alors un nouveau neurone ou groupe de neurones est ajouté et connecté au foyer. Un neurone est ajouté quand ![formula](https://render.githubusercontent.com/render/math?math=||d||>a_n)   avec ![formula](https://render.githubusercontent.com/render/math?math=a_n) ∈ ![formula](https://render.githubusercontent.com/render/math?math=R)
C'est à dire si la distance minimale entre l'entrée et les neurones dépasse le seuil ![formula](https://render.githubusercontent.com/render/math?math=a_n).


Élagage du réseau : On supprime les liens qui deviennent trop longs, c'est à dire soit ![formula](https://render.githubusercontent.com/render/math?math=a_r) le seuil, le lien entre le neurone ![formula](https://render.githubusercontent.com/render/math?math=i) et ![formula](https://render.githubusercontent.com/render/math?math=j) est supprimé si ![formula](https://render.githubusercontent.com/render/math?math=c_{ij}>a_r) . 

Quand un neurone n'a plus de lien, il est supprimé.

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

### Ajout des neurones

La première étape de la modélisation est la création du graphe et l'ajout de neurones. On définit alors la méthode `addNeuron` de la classe `Graph` prenant en paramètre un objet de la classe `Neuron`. <br>

On définit dans cette méthode 3 cas :
- Si le graphe est vide : le neurone prend comme label son index, et aucune liaison n'est alors créée.
- Si le graphe contient un seul neurone : on assigne au nouveau neurone le label du premier si la distance entre les deux est inférieure au seuil ![formula](https://render.githubusercontent.com/render/math?math=a_{n}), sinon son label est défini par son index. On crée ensuite la liaison entre les deux. (qui est ajouté aux deux neurones)

<br>

### Apprentissage et prédiction

#### Version de l'article

<br>

#### Version modifiée

<br>

### Affichage du graphe

Très vite, la nécessité d'avoir une représentation visuelle est devenu obligatoire. Ainsi, voici l'approche utilisée pour se faire.
Les données que nous avons à disposition sont les neurones avec leurs données ainsi que les liaisons entre eux. Nous calculerons toutes les distances nécessaires.
Le but est alors de générer les coordonnées des neurones pour pouvoir les plot avec Plotly. On se ramène à un problème purement mathématique, comment placer `n` points
en ne connaissant que la distance entre eux. Pour cela on va utiliser une méthode geométrique consistant à trouver l'intersection de n cercles, grâce à un système à n équations non linéaires.
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

<p align="center"><a href="#librairies"><img src="http://randojs.com/images/backToTopButton.png" alt="Haut de la page" height="29"/></a></p>

<br>

<br>

<p align="center">
  <a href="https://github.com/antonin-lfv" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97302854-e484da80-1859-11eb-9374-5b319ca51197.png" title="GitHub" width="40" height="40"></a>
  <a href="https://www.linkedin.com/in/antonin-lefevre-565b8b141" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303444-b2c04380-185a-11eb-8cfc-864c33a64e4b.png" title="LinkedIn" width="40" height="40"></a>
  <a href="mailto:antoninlefevre45@icloud.com" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303543-cec3e500-185a-11eb-8adc-c1364e2054a9.png" title="Mail" width="40" height="40"></a>
</p>


---------------------------

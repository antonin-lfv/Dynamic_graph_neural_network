import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
import math
import random
import json
from fastdist import fastdist
from typing import List
from scipy.optimize import fsolve
import pandas as pd
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')


class ConstGraph_article:
    # avec fct racines et cosinus
    INPUT_SIZE_CONFIG_1 = 100
    # avec les signaux
    INPUT_SIZE_CONFIG_2 = 300


class ConstGraph_custom:
    INPUT_SIZE = 250


class ConstThreshold_article:
    """La config 1 correspond à la classification des fonctions racines et sinus
    et la config 2 à la classification des signaux"""
    # altération du foyer (plus il est grand, plus il est modifié)
    # bv_config1 = 0.10
    bv = 0.90
    # altération des voisins du foyer (plus il est grand, plus ils sont modifiés)
    # bc_config1 = 0.10
    bc = 0.90
    # altération des liaisons du foyer (plus il est grand, plus elle est modifiée)
    # bl_config1 = 0.50
    bl = 0.90
    # seuil après lequel suppression des liens suite à altération
    # ar_config1 = 150
    ar = 120
    # seuil après lequel le neurone est seulement connecté au foyer
    # an_config1 = 100
    an = 30


class ConstThreshold_custom:
    # altération du foyer (plus il est grand, plus il est modifié)
    bv = 0.10
    # altération des voisins du foyer (plus il est grand, plus ils sont modifiés)
    bc = 0.10
    # altération des liaisons du foyer (plus il est grand, plus elle est modifiée)
    bl = 0.50
    # seuil après lequel suppression des liens suite à altération
    ar = 150
    # seuil après lequel le neurone est seulement connecté au foyer
    an = 100


class ConstPlotly:
    xaxis = dict(showgrid=False, zeroline=False,
                 showticklabels=False)
    yaxis = dict(showgrid=False, zeroline=False,
                 showticklabels=False)
    transparent_color = 'rgba(0,0,0,0)'

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
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')


class ConstGraph_article:
    # avec fct racines et cosinus
    INPUT_SIZE_CONFIG_1 = 100
    # avec les signaux
    INPUT_SIZE_CONFIG_2 = 250


class ConstGraph_custom:
    INPUT_SIZE = 250


class ConstThreshold_article:
    # altération du foyer (plus il est grand, plus il est modifié)
    bv_config1 = 0.10
    bv_config2 = 0.90
    # altération des voisins du foyer (plus il est grand, plus ils sont modifiés)
    bc_config1 = 0.10
    bc_config2 = 0.15
    # altération des liaisons du foyer (plus il est grand, plus elle est modifiée)
    bl_config1 = 0.50
    bl_config2 = 0.30
    # seuil après lequel suppression des liens suite à altération
    ar_config1 = 150
    ar_config2 = 150
    # seuil après lequel le neurone est seulement connecté au foyer
    an_config1 = 100
    an_config2 = 90


class ConstPlotly:
    xaxis = dict(showgrid=False, zeroline=False,
                 showticklabels=False)
    yaxis = dict(showgrid=False, zeroline=False,
                 showticklabels=False)
    transparent_color = 'rgba(0,0,0,0)'

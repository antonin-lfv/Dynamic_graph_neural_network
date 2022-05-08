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


class ConstThreshold_config1_article:
    # altération du foyer
    bv = 0.90
    # altération des voisins du foyer (taux)
    bc = 0.95
    # altération des liaisons du foyer
    bl = 0.90
    # seuil après lequel suppression des liens suite à altération
    ar = 150
    # seuil après lequel le neurone est seulement connecté au foyer
    an = 90


class ConstPlotly:
    xaxis = dict(showgrid=False, zeroline=False,
                 showticklabels=False)
    yaxis = dict(showgrid=False, zeroline=False,
                 showticklabels=False)
    transparent_color = 'rgba(0,0,0,0)'

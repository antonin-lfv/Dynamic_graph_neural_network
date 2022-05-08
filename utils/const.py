import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
import math
import json
from fastdist import fastdist
from typing import List
from scipy.optimize import fsolve
import pandas as pd
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')


class ConstGraph:
    INPUT_SIZE = 50


class ConstThreshold:
    # altération du foyer
    bv = 0.55
    # altération des voisins du foyer
    bc = 0.55
    # altération des liaisons du foyer
    bl = 0.95
    # seuil après lequel suppression des liens suite à altération
    ar = 100
    # seuil après lequel le neurone est seulement connecté au foyer
    an = 50


class ConstThreshold_v2:
    # altération du foyer
    bv = 0.85
    # altération des voisins du foyer
    bc = 0.65
    # altération des liaisons du foyer
    bl = 0.95
    # seuil après lequel suppression des liens suite à altération
    ar = 80
    # seuil après lequel le neurone est seulement connecté au foyer
    an = 90


class ConstPlotly:
    xaxis = dict(showgrid=False, zeroline=False,
                 showticklabels=False)
    yaxis = dict(showgrid=False, zeroline=False,
                 showticklabels=False)
    transparent_color = 'rgba(0,0,0,0)'
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
from scipy.fft import fft, fftfreq
import pywt
import librosa
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')


class ConstGraph_article:
    # avec fct racines et cosinus
    INPUT_SIZE_CONFIG_1 = 100
    # avec les signaux
    INPUT_SIZE_CONFIG_2 = 300
    # avec les FFT
    INPUT_SIZE_CONFIG_3 = 200


class ConstThreshold_article:
    """La config 1 correspond à la classification des fonctions racines et sinus
    et la config 2 à la classification des signaux"""
    # altération du foyer (plus il est grand, plus il est modifié)
    # bv_config1 = 0.10
    # bv_config2 = 0.70
    # bv_config3 = 0.20
    bv = 0.30
    # altération des voisins du foyer (plus il est grand, plus ils sont modifiés)
    # bc_config1 = 0.10
    # bc_config2 = 0.70
    # bc_config3 = 0.20
    bc = 0.20
    # altération des liaisons du foyer (plus il est grand, plus elle est modifiée)
    # bl_config1 = 0.50
    # bl_config2 = 0.70
    # bl_config3 = 0.20
    bl = 0.20
    # seuil après lequel suppression des liens suite à altération
    # ar_config1 = 150
    # ar_config2 = 100
    # ar_config3 = 30
    ar = 30
    # seuil après lequel le neurone est seulement connecté au foyer
    # an_config1 = 100
    # an_config2 = 35
    # an_config3 = 6.5
    an = 6.5


class ConstPlotly:
    xaxis = dict(showgrid=False, zeroline=False,
                 showticklabels=False)
    yaxis = dict(showgrid=False, zeroline=False,
                 showticklabels=False)
    transparent_color = 'rgba(0,0,0,0)'

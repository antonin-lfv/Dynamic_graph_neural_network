import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
import math as m
import json
from fastdist import fastdist
from typing import List


class ConstGraph:
    INPUT_SIZE = 50


class ConstThreshold:
    # altération du foyer
    bv = 0.1
    # altération des voisins du foyer
    bc = 0.05
    # seuil après lequel suppression des liens suite à altération
    ar = 80
    # seuil après lequel le neurone est seulement connecté au foyer
    an = 30


class ConstPlotly:
    xaxis = dict(showgrid=False, zeroline=False,
                 showticklabels=False)
    yaxis = dict(showgrid=False, zeroline=False,
                 showticklabels=False)
    transparent_color = 'rgba(0,0,0,0)'
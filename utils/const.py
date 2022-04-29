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
    bv = ...
    bc = ...
    seuilNouveauLabel = 10


class ConstPlotly:
    xaxis = dict(showgrid=False, zeroline=False,
                 showticklabels=False)
    yaxis = dict(showgrid=False, zeroline=False,
                 showticklabels=False)
    transparent_color = 'rgba(0,0,0,0)'
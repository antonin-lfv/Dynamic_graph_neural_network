import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
import math
import time
import itertools
import random
import json
from fastdist import fastdist
from typing import List, Callable
from scipy.optimize import fsolve
import pandas as pd
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pickle
from colored import fg
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

# ===== Config détails
# bv : altération du foyer (plus il est grand, plus il est modifié)
# bc : altération des voisins du foyer (plus il est grand, plus ils sont modifiés)
# bl : altération des liaisons du foyer (plus il est grand, plus elle est modifiée)
# ar : seuil après lequel suppression des liens suite à altération
# an : seuil après lequel le neurone est seulement connecté au foyer


class ConstPlotly:
    xaxis = dict(showgrid=False, zeroline=False,
                 showticklabels=False)
    yaxis = dict(showgrid=False, zeroline=False,
                 showticklabels=False)
    transparent_color = 'rgba(0,0,0,0)'

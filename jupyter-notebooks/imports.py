import numpy as np
import pandas as pd
import time
import random
import csv
import copy as cp
import heapq
import pickle
import cmcrameri.cm as cmc
from scipy.stats import norm

from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis
from PCAfold import reconstruction
from PCAfold import utilities
from PCAfold.styles import *

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

from sklearn import __version__ as sklearn_version
from scipy import __version__ as scipy_version
from PCAfold import __version__ as PCAfold_version

print('numpy==' + np.__version__)
print('pandas==' + pd.__version__)
print('PCAfold==' + PCAfold_version)
print('scipy==' + scipy_version)
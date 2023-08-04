import numpy as np
import pandas as pd
import time
import random
import csv
import os
import copy as cp
import heapq
import pickle
import cmcrameri.cm as cmc
from scipy.stats import norm
from scipy.integrate import odeint
from scipy.interpolate import RBFInterpolator

from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis
from PCAfold import reconstruction
from PCAfold import utilities
from PCAfold.styles import *
from PCAfold import QoIAwareProjection
from PCAfold import ANN

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
from  matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})

# from sklearn import __version__ as sklearn_version
# from scipy import __version__ as scipy_version
# from PCAfold import __version__ as PCAfold_version
# from platform import python_version

# print('Python==' + python_version())
# print()
# print('numpy==' + np.__version__)
# print('pandas==' + pd.__version__)
# print('scipy==' + scipy_version)
# print('scikit-learn==' + sklearn_version)
# print('tensorflow==' + tf.__version__)
# print('keras==' + keras.__version__)
# print('PCAfold==' + PCAfold_version)

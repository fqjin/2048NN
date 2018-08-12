from board import *
# includes SIZE, DIMENSIONS, SIZE_SQRD
# includes Board class
# includes randrange, randint, numpy as np


import tensorflow as tf
import keras
from keras.layers import Activation, Dense, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.regularizers import l2
# Suppress Tensorflow build from source messages
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL']='2'











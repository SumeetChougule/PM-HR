import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as tfl
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
from tensorflow.keras import datasets, layers, models, losses

import tensorflow_probability as tfp
tfd = tfp.distributions

import flowpm
from astropy.cosmology import Planck15
from flowpm import linear_field, lpt_init, nbody, cic_paint, cic_readout
from flowpm.utils import r2c3d, c2r3d
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline

# MODEL

def FC(input_shape):
  input_data = tf.keras.Input(shape = input_shape)
  cnn1   = tfl.Conv3D(filters = 8, kernel_size= 3, strides=(1,1,1), padding='same', activation= 'relu',data_format='channels_last')(input_data)
  cnn2   = tfl.Conv3D(filters = 4, kernel_size= 3, strides=(1,1,1), padding='same', activation= 'relu',data_format='channels_last')(cnn1)
  p_pos  = cic_readout_features( cnn2 , pos)
  p_p    = tf.squeeze(p_pos)
  MLP1   = tfl.Dense(64, activation = 'relu')(p_p)
  MLP2   = tfl.Dense(32, activation = 'relu')(MLP1)
  out    = tfl.Dense( 1, activation = 'linear')(MLP2)

  model = tf.keras.Model(inputs = input_data, outputs = out)
  return model

FC_model = FC((32,32,32,1))

FC_model.compile(optimizer='adam', loss=losses.MeanSquaredError())

FC_model.fit(ip_cnn, f_diff, epochs=2)

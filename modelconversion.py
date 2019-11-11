from keras.models import Model, load_model, model_from_json
from keras.layers import Input, Dense, Activation, TimeDistributed, Bidirectional, Dropout, CuDNNLSTM, BatchNormalization, Conv1D, Conv2D, Cropping2D
from keras.optimizers import RMSprop
import numpy as np
import scipy.signal
import postprocess_labels
from sewa_data_continued import load_SEWA, load_annotations
from ccc import compute_ccc
from ccc import ccc_loss_1, ccc_loss_2, ccc_loss_3
from numpy.random import seed
from tensorflow import set_random_seed
import glob
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import keras.losses
from matplotlib import pyplot as plt

def convert_to_stdf(model):
    for layer in model.layers:
        if 'input' in layer.name:
            inputs= Input(shape=19+(1,))
            curL = inputs
        elif 'conv1d' in layer.name:
            curL = Conv1D(\
            filters=1, \
            kernel_size=[s.value for s in layer.weights[0].shape][:-1], \
            strides=(1,1),padding='same',\
            kernel_initializer=keras.initializers.Constant(layer.get_weights()[0]),\
            bias_initializer=keras.initializers.Constant(layer.get_weights()[1])\
            )(curL)
            curL = Cropping2D(cropping=((0, 0), (18, 18)),)(curL)
        elif 'dropout' in layer.name:
            curL = Dropout(rate=dropout)(curL)
        elif 'time_distributed' in layer.name:
            out=[]
            for n in range(1):
                outn = TimeDistributed(Dense(1,\
                kernel_initializer=keras.initializers.Constant(layer.get_weights()[0]),\
                bias_initializer=keras.initializers.Constant(layer.get_weights()[1])\
                ))(curL)
                outn = Activation(final_activation)(outn)
                out.append(outn)
    opmodel = Model(inputs=inputs, outputs=out)
    opmodel2 = Model(inputs=inputs, outputs=curL)
    return opmodel

def convert_to_2d(model,dropout=0.2,final_activation='linear'):
    for layer in model.layers:
        if 'input' in layer.name:
            inputs= Input(shape=layer.input_shape[1:]+(1,))
            curL = inputs
        elif 'conv1d' in layer.name:
            curL = Conv2D(\
            filters=1, \
            kernel_size=[s.value for s in layer.weights[0].shape][:-1], \
            strides=(1,1),padding='same',\
            kernel_initializer=keras.initializers.Constant(layer.get_weights()[0]),\
            bias_initializer=keras.initializers.Constant(layer.get_weights()[1])\
            )(curL)
            curL = Cropping2D(cropping=((0, 0), (18, 18)),)(curL)
        elif 'dropout' in layer.name:
            curL = Dropout(rate=dropout)(curL)
        elif 'time_distributed' in layer.name:
            out=[]
            for n in range(1):
                outn = TimeDistributed(Dense(1,\
                kernel_initializer=keras.initializers.Constant(layer.get_weights()[0]),\
                bias_initializer=keras.initializers.Constant(layer.get_weights()[1])\
                ))(curL)
                outn = Activation(final_activation)(outn)
                out.append(outn)
    opmodel = Model(inputs=inputs, outputs=out)
    opmodel2 = Model(inputs=inputs, outputs=curL)
    return opmodel

def convert_to_2dpure(model,dropout=0.2,final_activation='linear'):
    for lid,layer in enumerate(model.layers):
        if 'input' in layer.name:
            inputs= Input(shape=layer.input_shape[1:]+(1,))
            curL = inputs
        elif 'conv1d' in layer.name:
            if 'time' in model.layers[lid+2].name:
                print(layer.get_weights()[0].shape, model.layers[lid+2].get_weights()[0].flatten())
                print(layer.get_weights()[1].shape, model.layers[lid+2].get_weights()[1].flatten())
                new_weight = layer.get_weights()[0]*model.layers[lid+2].get_weights()[0].flatten()
                new_bias = layer.get_weights()[1]*model.layers[lid+2].get_weights()[0].flatten()+model.layers[lid+2].get_weights()[1].flatten()
                curL = Conv2D(\
                filters=1, \
                kernel_size=[s.value for s in layer.weights[0].shape][:-1], \
                strides=(1,1),padding='same',\
                kernel_initializer=keras.initializers.Constant(new_weight),\
                bias_initializer=keras.initializers.Constant(new_bias)\
                )(curL)
                curL = Cropping2D(cropping=((0, 0), (18, 18)),)(curL)
        # elif 'dropout' in layer.name:
        #     curL = Dropout(rate=dropout)(curL)
        # elif 'time_distributed' in layer.name:
        #     out=[]
        #     for n in range(1):
        #         outn = TimeDistributed(Dense(1,\
        #         kernel_initializer=keras.initializers.Constant(layer.get_weights()[0]),\
        #         bias_initializer=keras.initializers.Constant(layer.get_weights()[1])\
        #         ))(curL)
        #         outn = Activation(final_activation)(outn)
        #         out.append(outn)
    opmodel = Model(inputs=inputs, outputs=[curL])
    opmodel2 = Model(inputs=inputs, outputs=curL)
    print(opmodel.summary())
    return opmodel

#opmodel=convert_to_2d(model)

import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16

import lucid.optvis.param as param
import lucid.optvis.transform as transform

from lucid4keras import prepare_model, keras_render_vis
from lucid4keras import objectives as keras_objectives

import matplotlib.pyplot as plt
# %matplotlib inline

import numpy as np
from ccc import ccc_loss_1, ccc_loss_2, ccc_loss_3
from keras.models import Model, load_model, model_from_json

modelFile          = '/home/vedhas/workspace/is2019_recurrence/results/8_oV_iv/m498_tr0.647_dv0.560_ts0.581_tc0.375.h5'

K.clear_session()
modelFile
model = load_model(modelFile, custom_objects={'ccc_loss_2': ccc_loss_2})
print([layer.name for layer in model.layers])
modelprep = prepare_model(model,layer_name="activation_3")
images = keras_render_vis(modelprep, 10)
plt.imshow(images[0])
plt.show()

from lucid.modelzoo.vision_models import Model
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16

import lucid.optvis.param as param
import lucid.optvis.transform as transform

from lucid4keras import prepare_model, keras_render_vis
from lucid4keras import objectives as keras_objectives

import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow as tf
import numpy as np
from ccc import ccc_loss_1, ccc_loss_2, ccc_loss_3
from keras.models import Model, load_model, model_from_json

modelFile          = '/home/vedhas/workspace/is2019_recurrence/results/12_oV_iv/m997_tr0.545_dv0.527_ts0.498_tc0.332.h5'
layer_name = 'conv1d_1'
filter_index = 0

# K.clear_session()
model = load_model(modelFile, custom_objects={'ccc_loss_2': ccc_loss_2})
layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_output = layer_dict[layer_name].output

loss = K.mean(layer_output[:, :, :, filter_index])
# compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# this function returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads])

import numpy as np

# we start from a gray image with some noise
input_img_data = np.random.random((1, 3, img_width, img_height)) * 20 + 128.
# run gradient ascent for 20 steps
for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step
from scipy.misc import imsave

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

img = input_img_data[0]
img = deprocess_image(img)
imsave('%s_filter_%d.png' % (layer_name, filter_index), img)

'''
print([layer.name for layer in model.layers])
# with tf.Graph().as_default() as graph, tf.Session() as sess:
with K.get_session().as_default():
    # images = tf.placeholder("float32", [None, 1768, 37], name="input")
    # model2 = Model.save(model, modelFile+"new.pb")
    model2 = Model.load(modelFile+"new.pb")
    render.render_vis(model2, "activation_3")
    plt.show()
'''

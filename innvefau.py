import innvestigate

import keras.backend as K
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow as tf
import numpy as np
from ccc import ccc_loss_1, ccc_loss_2, ccc_loss_3
from keras.models import Model, load_model, model_from_json

from matplotlib import pyplot as plt
modelFile          = '/home/vedhas/workspace/is2019_recurrence/results/12_oV_iv/m997_tr0.545_dv0.527_ts0.498_tc0.332.h5'
model = load_model(modelFile, custom_objects={'ccc_loss_2': ccc_loss_2})
print([layer.name for layer in model.layers])
filters, biases = model.layers[1].get_weights()
print(filters.shape)





sfaf
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
    # get the filter
    f = filters[:, :, i]
    plt.imshow(f, cmap='gray')
    plt.show()

# analyzer = innvestigate.create_analyzer("gradient", model)
'''
# https://github.com/albermax/innvestigate/issues/113
from keras import Sequential
from keras.layers import Dense, Conv1D, Embedding, GlobalMaxPooling1D
import numpy as np
import innvestigate

model = Sequential()
model.add(Embedding(input_dim=219, output_dim=8))
model.add(Conv1D(filters=64, kernel_size=8, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation=None))

#test
model.predict(np.random.randint(1,219, (1,100)))  # [[0.04913538 0.04234646]]

analyzer = innvestigate.create_analyzer('lrp.epsilon', model, neuron_selection_mode="max_activation", **{'epsilon': 1})
analyzer = innvestigate.create_analyzer('input_t_gradient', model)
analyzer.analyze(np.random.randint(1, 219, (1,100)))
'''

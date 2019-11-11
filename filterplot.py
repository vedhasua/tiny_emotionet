import innvestigate
import os
import keras.backend as K
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow as tf
import numpy as np
from ccc import ccc_loss_1, ccc_loss_2, ccc_loss_3
from keras.models import Model, load_model, model_from_json
import seaborn as sns
from matplotlib import pyplot as plt
modelFile          = '/home/vedhas/workspace/is2019_recurrence/DHCresults/2/m1000_tr0.687_dv0.608_ts0.625_tc0.420.h5'
# '/home/vedhas/workspace/is2019_recurrence/DHCresults/4/m1491_tr0.687_dv0.613_ts0.621_tc0.414.h5'
model = load_model(modelFile, custom_objects={'ccc_loss_2': ccc_loss_2})
print([layer.name for layer in model.layers])
filters, biases = model.layers[1].get_weights()
wdense, bdense = model.layers[3].get_weights()
wdense = wdense.flatten()
bdense = bdense.flatten()
print(filters.shape)
sns.heatmap(np.transpose(filters[:,:,0]))
plt.show()

feature_type_v='faus'
targets_avl='V'
fsuffix=targets_avl+"_v_{}".format(feature_type_v)

DeNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/De{}.npy'.format(fsuffix)
HuNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/Hu{}.npy'.format(fsuffix)
CnNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/Cn{}.npy'.format(fsuffix)

if os.path.isfile(DeNPY):
    trainX, trainY, develX, develY, testX, testY, origTrain, origDevel, origTest = np.load(DeNPY,allow_pickle=True, encoding = 'bytes')
if os.path.isfile(CnNPY):
    trainCX, trainCY, develCX, develCY, testCX, testCY, origCTrain, origCDevel, origCTest, origCrossC, crossXC, crossYC = np.load(CnNPY,allow_pickle=True, encoding = 'bytes')
if os.path.isfile(HuNPY):
    trainHX, trainHY, develHX, develHY, testHX, testHY, origHTrain, origHDevel, origHTest, origCross, crossX, crossY = np.load(HuNPY,allow_pickle=True, encoding = 'bytes')

plt.figure()
score=np.transpose(np.multiply(testCX[34,1350:1350+80,:],filters[:,:,0]))*wdense#+(biases.flatten()*wdense+bdense)/(80*37)
# plt.imshow(score)
sns.heatmap(score)
plt.show()

# normalize filter values to 0-1 so we can visualize them
# f_min, f_max = filters.min(), filters.max()
# filters = (filters - f_min) / (f_max - f_min)
ntsteps,nfeats,nfilters=filters.shape
# plot first few filters
# n_filters, ix = 6, 1
num_cols = 6
num_rows = 6 #int(math.ceil(tot_plots / float(num_cols)))
columnNames=[   'confidence_mean','confidence_std', 'AU01_mean','AU01_std', 'AU02_mean','AU02_std', 'AU04_mean','AU04_std',
                'AU05_mean','AU05_std', 'AU06_mean','AU06_std', 'AU07_mean','AU07_std', 'AU09_mean','AU09_std',
                'AU10_mean','AU10_std', 'AU12_mean','AU12_std', 'AU14_mean','AU14_std', 'AU15_mean','AU15_std',
                'AU17_mean','AU17_std', 'AU20_mean','AU20_std', 'AU23_mean','AU23_std', 'AU25_mean','AU25_std',
                'AU26_mean','AU26_std', 'AU45_mean','AU45_std', 'Turns'    ,'Culture',  'Arousal',  'Valence', 'Liking']
for idx in range(nfeats):
    print("{}:{}".format(columnNames[idx],np.median(filters[25:,idx,0])))

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(8, 10))
for idx, ax in enumerate(axes.flatten()):
    # for i in range(nfeats):
    ax.plot(filters[:,idx,0],label=columnNames[idx])
    ax.legend()

plt.show()
    # get the filter
    # f = filters[:, :, i]
    # plt.imshow(f, cmap='gray')
    # plt.show()

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
'''
confidence_mean:0.02069592848420143
confidence_std:0.0003188457922078669
AU01_mean:-0.010559577494859695
AU01_std:0.008166076615452766
AU02_mean:0.007076657842844725
AU02_std:-0.017698246985673904
AU04_mean:0.0032034432515501976
AU04_std:-0.0039812796749174595
AU05_mean:0.0008888975717127323
AU05_std:-0.01817190833389759
AU06_mean:0.012026654556393623
AU06_std:0.0006949214730411768
AU07_mean:0.0075859688222408295
AU07_std:-0.024745047092437744
AU09_mean:-0.0035023940727114677
AU09_std:-0.00934748537838459
AU10_mean:-0.008478161878883839
AU10_std:-0.0003787035238929093
AU12_mean:0.1448725163936615
AU12_std:-0.01410811860114336
AU14_mean:-0.029798515141010284
AU14_std:0.008726459927856922
AU15_mean:-0.011878572404384613
AU15_std:0.0035691047087311745
AU17_mean:0.01033693179488182
AU17_std:0.0235743410885334
AU20_mean:0.035397112369537354
AU20_std:0.01582670584321022
AU23_mean:-0.04549650102853775
AU23_std:-0.023455165326595306
AU25_mean:0.0070649972185492516
AU25_std:-0.04483116418123245
AU26_mean:-0.06919202208518982
AU26_std:0.018137071281671524
AU45_mean:-0.0036687462124973536
AU45_std:0.03953559696674347
Turns:0.013294939883053303
'''

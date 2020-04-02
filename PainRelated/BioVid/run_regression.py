
import os
import glob
from keras.models import Model
from keras.optimizers import RMSprop, Adam, Adagrad, Adadelta, Nadam
import numpy as np
import scipy.signal
from ccc import compute_ccc
from ccc import ccc_loss_1, ccc_loss_2, ccc_loss_3
from matplotlib import pyplot as plt

from numpy.random import seed
from tensorflow import set_random_seed
import pandas as pd
from fau_load import fau_csv_to_numpy, generate_model_cnn, get_loss
from standardise import standardise_3, reverse_rescale_3, scale_3

#what_to_test = [1, 5, 7, 8, 9, 11, 14, 16, 17, 22, 24, 26, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 41]

scaletype=1
perform_training = True
batch_size = 1024
max_num_epochs = 200
max_seq_len  = 138
num_cells_1  = [0, 0]
num_cells_2  = [0, 0]
num_cells_3  = [0, 0]
rfield = 24
num_cells_4  = [1, rfield]
last_specific = False
batch_norm    = False
final_activation = 'linear'
num_targets = 1
dropout = 0.3       #Not gonnna change anything, commented out for layer 4, the rest layers don't exist
learning_rate = 0.001
loss_function = 'ccc_2'
loss, loss_weights = [[get_loss(loss_function)], [1.0]]

#### DATA IMPORT #####

# fau and pspi imports 
fau_npy_load=True
if fau_npy_load:
    pspi_5d = np.load('pspi_5d.npy', allow_pickle=True)
    fau_5d = np.load('fau_5d.npy', allow_pickle=True)
else:
    fau_5d, pspi_5d = fau_csv_to_numpy()

fau_5d = fau_5d[:,:,:,:,2:]
num_subjs = fau_5d.shape[0]
num_pains = fau_5d.shape[1]
num_stims = fau_5d.shape[2]
num_tstep = fau_5d.shape[3]
num_feats = fau_5d.shape[4]
num_features = num_feats


train_subjs = np.arange(58,87)
valid_subjs = np.arange(29,58)
testt_subjs = np.arange(0,29)

trainX, trainY = fau_5d[train_subjs, : ], pspi_5d[train_subjs, : ]
validX, validY = fau_5d[valid_subjs, : ], pspi_5d[valid_subjs, : ], 
testtX, testtY = fau_5d[testt_subjs, : ], pspi_5d[testt_subjs, : ], 

trainX, validX, testtX = [ cur_array.reshape(-1,num_tstep,num_feats) for cur_array in [trainX, validX, testtX] ]
trainY, validY, testtY = [ cur_array.reshape(-1,num_tstep,1) for cur_array in [trainY, validY, testtY] ]

print([i.shape for i in [trainX, validX, testtX]+[trainY, validY, testtY] ])

seed=42
np.random.seed(seed)

apply_scaling = True
if apply_scaling:
    trainX, validX, testtX = scale_3(trainX, validX, testtX, scaletype)
    trainY, validY, testtY = scale_3(trainY, validY, testtY, scaletype)

noise = np.random.normal(0,0.01,max_seq_len*num_features).reshape((max_seq_len,-1))
trainX = trainX + noise


if perform_training:
    inputs, outputs = generate_model_cnn(max_seq_len, num_features, num_targets, num_cells_1, num_cells_2, num_cells_3, num_cells_4, batch_norm, last_specific, final_activation, dropout)
    optimizer = RMSprop(lr=learning_rate)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, sample_weight_mode=None)

    print(model.summary())

    model.fit(trainX, trainY, validation_data=(validX,validY), batch_size=batch_size, epochs=max_num_epochs, sample_weight=None)
    model.save("s{}b{}e{}r{}n{}.h5".format(seed,batch_size,max_num_epochs,rfield,num_feats))
    '''
    epoch = 1
    while epoch <= max_num_epochs:
        print("Iter: " + str(epoch))
        # Training iteration
        model.fit(trainX, trainY, validation_data=(validX,validY), batch_size=batch_size, epochs=epoch, sample_weight=None)
        epoch += 1
    '''
else:
    model=load_model(
                #"b{}e{}d{}.h5".format(batch_size,max_num_epochs,dropout), 
                #'s42b128e4000d0.0.h5',
            #'s0b128e8000d0.0r60.h5',
            's42b128e8000d0.0r24.h5',
                custom_objects={'ccc_loss_2':ccc_loss_2})

predict=model.predict(testtX)

test_ccc_list=[]
concat_test_predict=np.array([])
concat_test_ground=np.array([])
for cur_test_id, orig_test_id in enumerate(testt_subjs):
    if True: #orig_test_id not in [ 0,  2,  3,  4,  6, 10, 12, 13, 15, 18, 19, 20, 21, 23, 25, 27, 29, 39, 40, 43, 45, 47, 50, 51, 52, 54, 55, 57, 59]:
        cur_predict = predict[cur_test_id].flatten()
        cur_ground = testtY[cur_test_id].flatten()
        concat_test_predict = np.append(concat_test_predict,cur_predict)
        concat_test_ground  = np.append(concat_test_ground,cur_ground)
        cur_ccc = compute_ccc(cur_ground,cur_predict)
        test_ccc_list.append(cur_ccc)
concat_test_ccc = compute_ccc(concat_test_ground,concat_test_predict)
print(test_ccc_list,np.mean(test_ccc_list),concat_test_ccc)
'''
for test_id in what_to_test:
    plt.plot(predict[test_id])
    plt.plot(testtY[test_id])
    plt.title("{}: {}".format(test_id,facs_list[test_id]))
    plt.show()
'''

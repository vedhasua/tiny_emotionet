import os
import glob
from keras import regularizers
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Activation, TimeDistributed, Bidirectional, Dropout, CuDNNLSTM, BatchNormalization, Conv1D
from keras.optimizers import RMSprop, Adam, Adagrad, Adadelta, Nadam
import numpy as np
import scipy.signal
from ccc import compute_ccc
from ccc import ccc_loss_1, ccc_loss_2, ccc_loss_3
from matplotlib import pyplot as plt

from numpy.random import seed
from tensorflow import set_random_seed

what_to_test = [1, 5, 7, 8, 9, 11, 14, 16, 17, 22, 24, 26, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 41]

perform_training = True
batch_size = 128
max_num_epochs = 8000
max_seq_len  = 683
num_features = 14
num_cells_1  = [0, 0]
num_cells_2  = [0, 0]
num_cells_3  = [0, 0]
rfield = 1
num_cells_4  = [1, rfield]
last_specific = False
batch_norm    = False
final_activation = 'linear'
num_targets = 1
dropout = 0.0       #Not gonnna change anything, commented out for layer 4, the rest layers don't exist
learning_rate = 0.0008
loss_function = 'ccc_2'

facs_dir = '/home/vedhas/workspace/EmoPain/UNbC/Sequence_Labels/FACS/'
pspi_dir = '/home/vedhas/workspace/EmoPain/UNbC/Sequence_Labels/PSPI/'

facs_list = [ os.path.basename(cur_path) for cur_path in sorted(glob.glob(facs_dir+'*.csv')) ]
pspi_list = [ os.path.basename(cur_path) for cur_path in sorted(glob.glob(pspi_dir+'*.csv')) ]

feature_matrix  = np.zeros((200, max_seq_len, num_features ))
X,      Y       = [ np.zeros((200, max_seq_len, num_features )), np.zeros((200, max_seq_len, 1)) ]
# trainX, trainY  = [ np.zeros((200, max_seq_len, num_features )), np.zeros((200, max_seq_len, 1)) ]
# testX,  testY   = [ np.zeros((200, max_seq_len, num_features )), np.zeros((200, max_seq_len, 1)) ]

total_vector_count=[]   # Total number of vectors, i.e. sum(seq_lengths)
total_zero_count=[]     # Total number of zero-vectors, i.e. #vectors where all activations=0  
fully_zero_index=[]     # List of zero-sequences, i.e. #sequences where all are zero-vectors and/or all pspi=0.

for cur_seq_id, cur_seq_name in enumerate(facs_list):
    cur_facs = np.loadtxt(facs_dir + cur_seq_name, delimiter=',')
    cur_zero_count = cur_facs.shape[0]-np.count_nonzero(cur_facs[:, 1:].sum(axis=1))
    cur_pspi = np.loadtxt(pspi_dir + cur_seq_name)
    cur_pspi = np.expand_dims(cur_pspi,1)
    total_vector_count.append(cur_facs.shape[0])
    total_zero_count.append(cur_zero_count)
    X[cur_seq_id, :cur_facs.shape[0], :] = cur_facs[:, 1:]
    Y[cur_seq_id, :cur_pspi.shape[0], :] = cur_pspi
    if cur_zero_count==cur_facs.shape[0]:#np.mean(cur_pspi)==0:
        fully_zero_index.append(cur_seq_id)
    # X = np.vstack(X, cur_facs[:, 1:])
    # Y = np.vstack(Y, cur_pspi)
#print(facs_list)
#X=X[:,:,[1,2,3,4,5,-2]]
num_features=X.shape[-1]
print(total_zero_count)
print(fully_zero_index,len(fully_zero_index))
print(total_vector_count)

#print(np.cumsum(total_vector_count))
#print(facs_list[60:62],facs_list[121:123])

def generate_model_cnn(max_seq_len, num_features, num_targets, num_cells_1, num_cells_2, num_cells_3, num_cells_4, batch_norm, last_specific, final_activation, dropout=0.0):
    # Input
    inputs = Input(shape=(max_seq_len ,num_features))
    net = inputs

    if num_cells_1[0] > 0:
        # 1st layer
        net = Conv1D(num_cells_1[0], num_cells_1[1], strides=1, padding='same')(net)
        if batch_norm: net = BatchNormalization()(net)
        net = Activation('linear')(net)
        net = Dropout(rate=dropout)(net)

    # 2nd layer
    if num_cells_2[0] > 0:
        net = Conv1D(num_cells_2[0], num_cells_2[1], strides=1, padding='same')(net)
        if batch_norm: net = BatchNormalization()(net)
        net = Activation('linear')(net)
        net = Dropout(rate=dropout)(net)

    # 3rd layer
    if num_cells_3[0] > 0:
        net = Conv1D(num_cells_3[0], num_cells_3[1], strides=1, padding='same')(net)
        if batch_norm: net = BatchNormalization()(net)
        net = Activation('linear')(net)
        net = Dropout(rate=dropout)(net)

    if not last_specific:
        # 4th layer
        if num_cells_4[0] > 0:
            net = Conv1D(num_cells_4[0], num_cells_4[1], strides=1, padding='same', kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01))(net)
            if batch_norm: net = BatchNormalization()(net)
            #net = Activation(final_activation)(net)
            #net = Dropout(rate=dropout)(net)

        # outputs (& task-specific layers)
        out = []
        for n in range(num_targets):
            outn = TimeDistributed(Dense(1))(net)
            #outn = Activation(final_activation)(outn)
            out.append(outn)
    else:  # 4th layer mandatory!
        out = []
        for n in range(num_targets):
            net_part = Conv1D(num_cells_4[0], num_cells_4[1], strides=1, padding='same')(net)
            if batch_norm: net_part = BatchNormalization()(net_part)
            net_part = Activation('linear')(net_part)
            net_part = Dropout(rate=dropout)(net_part)
            #
            outn = TimeDistributed(Dense(1))(net_part)
            outn = Activation(final_activation)(outn)
            out.append(outn)

    return inputs, out

def get_loss(loss_function):
    if   loss_function=='ccc_1': loss = ccc_loss_1
    elif loss_function=='ccc_2': loss = ccc_loss_2  # not faster, maybe(!) better in terms of the result
    elif loss_function=='ccc_3': loss = ccc_loss_3
    elif loss_function=='mse':   loss = 'mean_squared_error'
    return loss


loss, loss_weights = [[get_loss(loss_function)], [1.0]]

trainsamples=np.arange(129,200)
validsamples=np.arange(61,129)
testtsamples=np.arange(0,61)

seed=42
np.random.seed(seed)
noise = np.random.normal(0,0.01,max_seq_len*num_features).reshape((max_seq_len,-1))
trainX, trainY = X[trainsamples,:], Y[trainsamples,:]
validX, validY = X[validsamples,:], Y[validsamples,:]
testtX, testtY = X[testtsamples,:], Y[testtsamples,:]

trainX = trainX + noise


if perform_training:
    inputs, outputs = generate_model_cnn(max_seq_len, num_features, num_targets, num_cells_1, num_cells_2, num_cells_3, num_cells_4, batch_norm, last_specific, final_activation, dropout)
    optimizer = RMSprop(lr=learning_rate)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, sample_weight_mode=None)

    print(model.summary())

    CCCs = np.empty((0, num_targets * 3))
    CCCs_orig = np.empty((0, num_targets * 3))
    CCCs_orig_seq = np.empty((0, num_targets * 3))
    CCCs_orig_pp = np.empty((0, num_targets * 2))
    model.fit(trainX, trainY, validation_data=(validX,validY), batch_size=batch_size, epochs=max_num_epochs, sample_weight=None)
    model.save("s{}b{}e{}d{}r{}.h5".format(seed,batch_size,max_num_epochs,dropout,rfield))
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
for cur_test_id, orig_test_id in enumerate(testtsamples):
    if orig_test_id not in [ 0,  2,  3,  4,  6, 10, 12, 13, 15, 18, 19, 20, 21, 23, 25, 27, 29, 39, 40, 43, 45, 47, 50, 51, 52, 54, 55, 57, 59]:
        cur_predict = predict[cur_test_id][:total_vector_count[orig_test_id]].flatten()
        cur_ground = testtY[cur_test_id][:total_vector_count[orig_test_id]].flatten()
        concat_test_predict = np.append(concat_test_predict,cur_predict)
        concat_test_ground  = np.append(concat_test_ground,cur_ground)
        cur_ccc = compute_ccc(cur_ground,cur_predict)
        test_ccc_list.append(cur_ccc)
concat_test_ccc = compute_ccc(concat_test_ground,concat_test_predict)
print(test_ccc_list,concat_test_ccc)
'''
for test_id in what_to_test:
    plt.plot(predict[test_id])
    plt.plot(testtY[test_id])
    plt.title("{}: {}".format(test_id,facs_list[test_id]))
    plt.show()
'''

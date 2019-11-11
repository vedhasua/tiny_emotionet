from keras.models import Model
from keras.layers import Input, Dense, Activation, TimeDistributed, Bidirectional, Dropout, CuDNNLSTM, BatchNormalization, Conv1D
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

def main(param_given=False, culture='German', eval_cross=True, modality='audio', get_turn_feature=True, uncertainty_target=False, invert_std=True, loss_unc='ccc_2', weight_unc=1.0, balance_weight=False, uncertainty_weight=False, batch_size=34, learning_rate=0.001, max_num_epochs=200, first_lstm=False, num_cells_1=128, num_cells_2=0, num_cells_3=0, num_cells_4=0, last_lstm=False, batch_norm=False, last_specific=False, comb_smoothing=False, bidirectional=False, dropout=0.0, final_activation='linear', loss_function='ccc_2', shift_sec=2.0, targets_avl='AVL', feature_type_a='funcegem', feature_type_v='func', window_size=4.0, xbow_cs=100, xbow_na=1, random_seed=0, add_noise=False, append_results_file='all_resultsDHC_mean.txt'):
    # Configuration
    if not param_given:
        culture            = 'German'
        eval_cross         = True
        modality           = 'video'
        get_turn_feature   = True
        uncertainty_target = False     #
        invert_std         = False     # Only relevant if use_uncertainty==True
        loss_unc           = 'ccc_2'
        weight_unc         = 1.0
        balance_weight     = False
        uncertainty_weight = False

        batch_size       = 34       #
        learning_rate    = 0.001    #
        max_num_epochs   = 2000     #
        first_lstm       = False    # only for LSTM models
        num_cells_1      = [0,15]   # if scalar: LSTM model
        num_cells_2      = [0,20]  # CNN model requires list [num_filters, filter_length]: =[0,0] means no 2nd layer
        num_cells_3      = [0,30]   # CNN model requires list [num_filters, filter_length]: =[0,0] means no 2nd layer
        num_cells_4      = [1,80]  # CNN model requires list [num_filters, filter_length]: =[0,0] means no 2nd layer
        last_lstm        = False    # only for LSTM models
        batch_norm       = False    #
        last_specific    = False    #
        comb_smoothing   = False    # only for LSTM models
        bidirectional    = False    # only for LSTM models
        dropout          = 0.0      # dropout - same for all
        final_activation = 'linear' # 'linear' or 'tanh'
        loss_function    = 'ccc_2'  #

        shift_sec        = 2.8      #PARAM - 0.05 opt for window size 0.1, uni-directional LSTM
        targets_avl      = 'V' #USE ONLY ONE!!!     #
        feature_type_a   = 'funcegem'   # see sewa_data.py
        feature_type_v   = 'faus'   # see sewa_data.py
        window_size      = 0.1      # only relevant for egem, boaw, func
        xbow_cs          = 1000      # Codebook size, only relevant for boaw
        xbow_na          = 10        # Number of assignments, only relevant for boaw
        random_seed      = 0        #
        add_noise        = False    # TODO: Noise layer (not yet implemented)

        fsuffix           = targets_avl+''
        if 'audio' in modality:
            fsuffix += "_a_{}".format(feature_type_a)
        if 'video' in modality:
            fsuffix += "_v_{}".format(feature_type_v)

        # DeNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/De{}.npy'.format(fsuffix)
        # HuNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/Hu{}.npy'.format(fsuffix)
        # CnNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/Cn{}.npy'.format(fsuffix)
        DeNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/DeAVL_v_faus.npy'
        HuNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/HuAVL_v_faus.npy'
        CnNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/CnAVL_v_faus.npy'

    # set seed to be able to reproduce results (does not really work as it depends also on the hardware)
    seed(random_seed+1)
    set_random_seed(random_seed+2)

    if feature_type_a=='raw':
        inst_per_sec   = 100
        factor_tr_orig = 10  # factor between labels for training and original labels (in terms of fps)
    elif feature_type_v=='raw':
        inst_per_sec   = 50
        factor_tr_orig = 5  # factor between labels for training and original labels (in terms of fps)
    else:
        inst_per_sec   = 10
        factor_tr_orig = 1

    shift = int(np.round(shift_sec*inst_per_sec))

    # Targets
    targets = []
    if targets_avl.find('A')>=0: targets.append(0)
    if targets_avl.find('V')>=0: targets.append(1)
    if targets_avl.find('L')>=0: targets.append(2)

    # Load SEWA data
    print('Loading SEWA database ...')
    if os.path.isfile(DeNPY):
        trainX, trainY, develX, develY, testX, testY, origTrain, origDevel, origTest = np.load(DeNPY,allow_pickle=True)
        trainY, develY, testY, origTrain, origDevel, origTest = [[curList[targets[0]]] for curList in [trainY, develY, testY, origTrain, origDevel, origTest] ]
        trainX = trainX[:,:,np.arange(0,trainX.shape[2],2)]
        develX = develX[:,:,np.arange(0,develX.shape[2],2)]
        testX  = testX[:,:,np.arange(0,testX.shape[2],2)]

    else:
        assert False
        trainX, trainY, develX, develY, testX, testY, origTrain, origDevel, origTest = load_SEWA(culture=culture,
                                                                                             modality=modality,
                                                                                             mode_audio=feature_type_a,
                                                                                             mode_video=feature_type_v,
                                                                                             get_turn_feature=get_turn_feature,
                                                                                             window_size=window_size,
                                                                                             xbow_cs=xbow_cs,
                                                                                             xbow_na=xbow_na,
                                                                                             targets=targets)  # get also turn annotations
        np.save(DeNPY,[trainX, trainY, develX, develY, testX, testY, origTrain, origDevel, origTest])
        trainX = trainX[:,:,np.arange(0,trainX.shape[2],2)]
        develX = develX[:,:,np.arange(0,develX.shape[2],2)]
        testX  = testX[:,:,np.arange(0,testX.shape[2],2)]
    ## Cross
    if eval_cross:
        if os.path.isfile(HuNPY):
            trainHX, trainHY, develHX, develHY, testHX, testHY, origHTrain, origHDevel, origHTest, origCross, crossX, crossY = np.load(HuNPY,allow_pickle=True)
            # print(len(trainHY),len(develHY),len(testHY),len(origHTrain),len(origHDevel),len(origHTest))
            trainHY, develHY, testHY, origHTrain, origHDevel, origHTest, origCross, crossY = [[curList[targets[0]]] for curList in [trainHY, develHY, testHY, origHTrain, origHDevel, origHTest, origCross, crossY] ]
            trainHX = trainHX[:,:,np.arange(0,trainHX.shape[2],2)]
            develHX = develHX[:,:,np.arange(0,develHX.shape[2],2)]
            testHX  = testHX[:,:,np.arange(0,testHX.shape[2],2)]
            crossX  = crossX[:,:,np.arange(0,crossX.shape[2],2)]
        else:
            assert False
            trainHX, trainHY, develHX, develHY, testHX, testHY, origHTrain, origHDevel, origHTest = load_SEWA(culture='Hungarian',
                                                                                                          modality=modality,
                                                                                                          mode_audio=feature_type_a,
                                                                                                          mode_video=feature_type_v,
                                                                                                          get_turn_feature=get_turn_feature,
                                                                                                          window_size=window_size,
                                                                                                          xbow_cs=xbow_cs,
                                                                                                          xbow_na=xbow_na,
                                                                                                          targets=targets)  # get also turn annotations
            trainHX = trainHX[:,:,np.arange(0,trainHX.shape[2],2)]
            develHX = develHX[:,:,np.arange(0,develHX.shape[2],2)]
            testHX  = testHX[:,:,np.arange(0,testHX.shape[2],2)]
            crossX = np.concatenate((trainHX, develHX, testHX), axis=0)
            crossY = []
            for y1, y2, y3 in zip(trainHY, develHY, testHY):
                crossY.append(np.concatenate((y1, y2, y3), axis=0))
            origCross = []
            for o1, o2, o3 in zip(origHTrain, origHDevel, origHTest):
                orig = []
                orig.extend(o1)
                orig.extend(o2)
                orig.extend(o3)
                origCross.append(orig)
            np.save(HuNPY,[trainHX, trainHY, develHX, develHY, testHX, testHY, origHTrain, origHDevel, origHTest, origCross, crossX, crossY])

        if os.path.isfile(CnNPY):
            trainCX, trainCY, develCX, develCY, testCX, testCY, origCTrain, origCDevel, origCTest, origCrossC, crossXC, crossYC = np.load(CnNPY,allow_pickle=True)
            trainCY, develCY, testCY, origCTrain, origCDevel, origCTest, origCrossC, crossYC = [[curList[targets[0]]] for curList in [trainCY, develCY, testCY, origCTrain, origCDevel, origCTest, origCrossC, crossYC] ]
            trainCX = trainCX[:,:,np.arange(0,trainCX.shape[2],2)]
            develCX = develCX[:,:,np.arange(0,develCX.shape[2],2)]
            testCX  = testCX[:,:,np.arange(0,testCX.shape[2],2)]
            crossXC  = crossXC[:,:,np.arange(0,crossXC.shape[2],2)]
        else:
            assert False
            trainCX, trainCY, develCX, develCY, testCX, testCY, origCTrain, origCDevel, origCTest = load_SEWA(culture='Chinese',
                                                                                                          modality=modality,
                                                                                                          mode_audio=feature_type_a,
                                                                                                          mode_video=feature_type_v,
                                                                                                          get_turn_feature=get_turn_feature,
                                                                                                          window_size=window_size,
                                                                                                          xbow_cs=xbow_cs,
                                                                                                          xbow_na=xbow_na,
                                                                                                          targets=targets)  # get also turn annotations
            crossXC = np.concatenate((trainCX, develCX, testCX), axis=0)
            crossYC = []
            for y1, y2, y3 in zip(trainCY, develCY, testCY):
                crossYC.append(np.concatenate((y1, y2, y3), axis=0))
            origCrossC = []
            for o1C, o2C, o3C in zip(origCTrain, origCDevel, origCTest):
                origC = []
                origC.extend(o1C)
                origC.extend(o2C)
                origC.extend(o3C)
                origCrossC.append(origC)
            np.save(CnNPY,[trainCX, trainCY, develCX, develCY, testCX, testCY, origCTrain, origCDevel, origCTest, origCrossC, crossXC, crossYC])


    ## Cross
    max_seq_len  = trainX.shape[1]  # same for all
    num_features = trainX.shape[2]  # same for all X
    num_targets  = len(targets)

    print('Shifting labels to the front for ' + str(shift_sec) + ' seconds.')
    for t in range(0, num_targets):
        trainY[t] = shift_annotations_to_front(trainY[t], shift)
        develY[t] = shift_annotations_to_front(develY[t], shift)
        testY[t]  = shift_annotations_to_front(testY[t],  shift)
        if eval_cross:
            crossY[t]  = shift_annotations_to_front(crossY[t],  shift)
            crossYC[t]  = shift_annotations_to_front(crossYC[t],  shift)

    # Create model
    if not isinstance(num_cells_1,list):
        inputs, outputs = generate_model_lstm(max_seq_len, num_features, num_targets, first_lstm, bidirectional, num_cells_1, num_cells_2, num_cells_3, num_cells_4, last_lstm, batch_norm, last_specific,
                                             final_activation, dropout, comb_smoothing)
    else:
        inputs, outputs = generate_model_cnn(max_seq_len, num_features, num_targets, num_cells_1, num_cells_2, num_cells_3, num_cells_4, batch_norm, last_specific, final_activation, dropout)

    # Learner
    loss, loss_weights = get_hyperparameters(targets, loss_function, uncertainty_target, loss_unc, weight_unc)
    rmsprop = RMSprop(lr=learning_rate)
    # Initialise weighting
    sample_weight_mode = None
    sample_weight      = None
    if balance_weight or uncertainty_weight:
        sample_weight_mode = "Add back remove code"

    # create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=rmsprop, loss=loss, loss_weights=loss_weights, sample_weight_mode=sample_weight_mode)

    print(model.summary())

    CCCs          = np.empty((0,num_targets*3))
    CCCs_orig     = np.empty((0,num_targets*3))
    CCCs_orig_seq = np.empty((0,num_targets*3))
    CCCs_orig_pp  = np.empty((0,num_targets*2))
    if eval_cross:
        CCCs          = np.empty((0,num_targets*5))
        CCCs_orig     = np.empty((0,num_targets*5))
        CCCs_orig_seq = np.empty((0,num_targets*5))
        CCCs_orig_pp  = np.empty((0,num_targets*4))
    cccmax=0
    epoch = 1
    while epoch <= max_num_epochs:
        print("Iter: " + str(epoch))
        # Training iteration
        model.fit(trainX, trainY, batch_size=batch_size, initial_epoch=epoch-1, epochs=epoch, sample_weight=sample_weight)

        # Evaluate and save results (in terms of CCC)
        if eval_cross:
            _        , _        , _       , CCC_testT1, _             , _             , _            , CCC_orig_testT1, _                 , _                 , _                , CCC_orig_test_seqT1, _                , _               , CCC_orig_test_ppT1 = evaluate_all_cross(model, trainX, develX, testX, crossX, trainY, develY, testY, crossY, origTrain, origDevel, origTest, origCross, shift, factor_tr_orig, num_targets)
            CCC_train, CCC_devel, CCC_test, CCC_testT2, CCC_orig_train, CCC_orig_devel, CCC_orig_test, CCC_orig_testT2, CCC_orig_train_seq, CCC_orig_devel_seq, CCC_orig_test_seq, CCC_orig_test_seqT2, CCC_orig_devel_pp, CCC_orig_test_pp, CCC_orig_test_ppT2 = evaluate_all_cross(model, trainX, develX, testX, crossXC, trainY, develY, testY, crossYC, origTrain, origDevel, origTest, origCrossC, shift, factor_tr_orig, num_targets)
            CCCs          = np.append(CCCs,         [np.array([CCC_train,          CCC_devel,          CCC_test,          CCC_testT1, CCC_testT2]).flatten()],          axis=0)
            CCCs_orig     = np.append(CCCs_orig,    [np.array([CCC_orig_train,     CCC_orig_devel,     CCC_orig_test,     CCC_orig_testT1, CCC_orig_testT2]).flatten()],     axis=0)
            CCCs_orig_seq = np.append(CCCs_orig_seq,[np.array([CCC_orig_train_seq, CCC_orig_devel_seq, CCC_orig_test_seq, CCC_orig_test_seqT1, CCC_orig_test_seqT2]).flatten()], axis=0)
            CCCs_orig_pp  = np.append(CCCs_orig_pp, [np.array([CCC_orig_devel_pp,  CCC_orig_test_pp,   CCC_orig_test_ppT1, CCC_orig_test_ppT2 ]).flatten()], axis=0)
            #####
            #CCCs          = np.append(CCCs,         [np.array([CCC_train,          CCC_devel,          CCC_test,          CCC_testT2]).flatten()],          axis=0)
            #CCCs_orig     = np.append(CCCs_orig,    [np.array([CCC_orig_train,     CCC_orig_devel,     CCC_orig_test,     CCC_orig_testT2]).flatten()],     axis=0)
            #CCCs_orig_seq = np.append(CCCs_orig_seq,[np.array([CCC_orig_train_seq, CCC_orig_devel_seq, CCC_orig_test_seq, CCC_orig_test_seqT2]).flatten()], axis=0)
            #CCCs_orig_pp  = np.append(CCCs_orig_pp, [np.array([CCC_orig_devel_pp,  CCC_orig_test_pp,   CCC_orig_test_ppT2                    ]).flatten()], axis=0)

            if CCC_orig_devel==cccmax:
                cccmax=CCC_orig_devel
                model.save("m{}_tr{:.3f}_dv{:.3f}_ts{:.3f}_tc{:.3f}.h5".format(epoch, CCC_orig_train[0], CCC_orig_devel[0], CCC_orig_test[0], CCC_orig_testT1[0]))
                lastsaved=epoch
            if CCC_orig_devel>cccmax:
                cccmax=CCC_orig_devel
                model.save("m{}_tr{:.3f}_dv{:.3f}_ts{:.3f}_tc{:.3f}.h5".format(epoch, CCC_orig_train[0], CCC_orig_devel[0], CCC_orig_test[0], CCC_orig_testT1[0]))
                try:
                    os.remove(glob.glob("m{}_*".format(lastsaved))[0])
                except:
                    pass
                lastsaved=epoch
            print("CCC train: {:.3f},{:.3f},{:.3f}, ".format(CCC_train[0],CCC_orig_train[0],CCC_orig_train_seq[0]) + \
                  "CCC devel: {:.3f},{:.3f},{:.3f},{:.3f} ".format(CCC_devel[0],CCC_orig_devel[0],CCC_orig_devel_seq[0],CCC_orig_devel_pp[0]) + \
                  "CCC test:  {:.3f},{:.3f},{:.3f},{:.3f} ".format(CCC_test[0], CCC_orig_test[0], CCC_orig_test_seq[0], CCC_orig_test_pp[0] ) + \
                "CCC testT1: {:.3f},{:.3f},{:.3f},{:.3f} ".format(CCC_testT1[0],CCC_orig_testT1[0],CCC_orig_test_seqT1[0],CCC_orig_test_ppT1[0])+ \
                "CCC testT2: {:.3f},{:.3f},{:.3f},{:.3f} ".format(CCC_testT2[0],CCC_orig_testT2[0],CCC_orig_test_seqT2[0],CCC_orig_test_ppT2[0]))
        else:
            CCC_train, CCC_devel, CCC_test, CCC_orig_train, CCC_orig_devel, CCC_orig_test, CCC_orig_train_seq, CCC_orig_devel_seq, CCC_orig_test_seq, CCC_orig_devel_pp, CCC_orig_test_pp = evaluate_all(model, trainX, develX, testX, trainY, develY, testY, origTrain, origDevel, origTest, shift, factor_tr_orig, num_targets)
            CCCs          = np.append(CCCs,         [np.array([CCC_train,          CCC_devel,          CCC_test]).flatten()],          axis=0)
            CCCs_orig     = np.append(CCCs_orig,    [np.array([CCC_orig_train,     CCC_orig_devel,     CCC_orig_test]).flatten()],     axis=0)
            CCCs_orig_seq = np.append(CCCs_orig_seq,[np.array([CCC_orig_train_seq, CCC_orig_devel_seq, CCC_orig_test_seq]).flatten()], axis=0)
            CCCs_orig_pp  = np.append(CCCs_orig_pp, [np.array([CCC_orig_devel_pp,  CCC_orig_test_pp]).flatten()],  axis=0)
            print("CCC train: " + str(np.round(CCC_train*1000)/1000) + str(np.round(CCC_orig_train*1000)/1000) + str(np.round(CCC_orig_train_seq*1000)/1000))
            print("CCC devel: " + str(np.round(CCC_devel*1000)/1000) + str(np.round(CCC_orig_devel*1000)/1000) + str(np.round(CCC_orig_devel_seq*1000)/1000) + str(np.round(CCC_orig_devel_pp*1000)/1000))
            print("CCC test:  " + str(np.round(CCC_test*1000)/1000)  + str(np.round(CCC_orig_test*1000)/1000)  + str(np.round(CCC_orig_test_seq*1000)/1000)  + str(np.round(CCC_orig_test_pp*1000)/1000) )
        epoch += 1  # Increase epoch counter

    # Get best result (optimised on devel)
    indexes_devel = range(num_targets*1, num_targets*2)  # Get development indexes
    indexes_test  = range(num_targets*2, num_targets*3)  # Get test indexes
    if eval_cross: indexes_test = range(num_targets*2, num_targets*4)  # Get test indexes
    iter_opt     = np.argmax(CCCs_orig[:,indexes_devel], axis=0)   # returns array
    res_dev_max  = np.round(CCCs_orig[iter_opt,indexes_devel]*1000)/1000
    res_test_max = np.round(CCCs_orig[iter_opt,indexes_test]*1000)/1000
    res_all      = np.concatenate((res_dev_max,res_test_max), axis=0)

    # Get best result after postprocessing (optimised on postprocessed devel)
    indexes_devel = range(num_targets*0,num_targets*1)  # Get development indexes - no train here!
    indexes_test  = range(num_targets*1,num_targets*2)  # Get test indexes - no train here!
    if eval_cross: indexes_test = range(num_targets*1, num_targets*3)  # Get test indexes - no train here!
    #iter_opt: Use the ones optimised on the not-postprocessed results - postprocessing might add a bias to devel
    res_dev_max  = np.round(CCCs_orig_pp[iter_opt,indexes_devel]*1000)/1000
    res_test_max = np.round(CCCs_orig_pp[iter_opt,indexes_test]*1000)/1000
    res_all_pp   = np.concatenate((res_dev_max,res_test_max), axis=0)

    # Append (global) results file
    config = [culture, eval_cross, modality, get_turn_feature, uncertainty_target, invert_std, loss_unc, weight_unc, balance_weight, balance_weight, uncertainty_weight, batch_size, learning_rate, first_lstm, num_cells_1, num_cells_2, num_cells_3, num_cells_4, last_lstm, batch_norm, last_specific, comb_smoothing, bidirectional, dropout, final_activation, loss_function, shift_sec, targets_avl, feature_type_a, feature_type_v, window_size, random_seed, add_noise, max_num_epochs, iter_opt]
    config.append('results')
    config.extend(res_all)
    config.append('postproc')
    config.extend(res_all_pp)
    append_csv(append_results_file, config)


def get_hyperparameters(targets, loss_function, uncertainty_target=False, loss_unc=False, weight_unc=0.0):
    if uncertainty_target:
        real_targets = len(targets) // 2
    else:
        real_targets = len(targets)
    loss = []
    loss_weight = []
    for _ in range(real_targets):
        loss.append(get_loss(loss_function))
        loss_weight.append(1.)
    if uncertainty_target:
        for _ in range(real_targets):
            loss.append(get_loss(loss_unc))
            loss_weight.append(weight_unc)
    return loss, loss_weight


def get_loss(loss_function):
    if   loss_function=='ccc_1': loss = ccc_loss_1
    elif loss_function=='ccc_2': loss = ccc_loss_2  # not faster, maybe(!) better in terms of the result
    elif loss_function=='ccc_3': loss = ccc_loss_3
    elif loss_function=='mse':   loss = 'mean_squared_error'
    return loss


def generate_model_lstm(max_seq_len, num_features, num_targets, first_lstm, bidirectional, num_cells_1, num_cells_2, num_cells_3, num_cells_4, last_lstm, batch_norm, last_specific, final_activation, dropout=0.0, comb_smoothing=False):
    # Input
    inputs = Input(shape=(max_seq_len,num_features))
    net = inputs

    # 1st layer
    if first_lstm:
        if bidirectional: net = Bidirectional(CuDNNLSTM(num_cells_1, return_sequences=True))(net)
        else:             net = CuDNNLSTM(num_cells_1, return_sequences=True)(net)
    else:
        net = TimeDistributed(Dense(num_cells_1))(net)
    if batch_norm: net = BatchNormalization()(net)
    net = Activation('tanh')(net)  # to be consistent with lstm
    net = Dropout(rate=dropout)(net)

    # 2nd layer
    if num_cells_2 > 0:
        if bidirectional: net = Bidirectional(CuDNNLSTM(num_cells_2, return_sequences=True))(net)
        else:             net = CuDNNLSTM(num_cells_2, return_sequences=True)(net)
        if batch_norm: net = BatchNormalization()(net)
        net = Activation('tanh')(net)
        net = Dropout(rate=dropout)(net)

    # 3rd layer
    if num_cells_3 > 0:
        if bidirectional: net = Bidirectional(CuDNNLSTM(num_cells_3, return_sequences=True))(net)
        else:             net = CuDNNLSTM(num_cells_3, return_sequences=True)(net)
        if batch_norm: net = BatchNormalization()(net)
        net = Activation('tanh')(net)
        net = Dropout(rate=dropout)(net)

    if not last_specific:
        # 4th layer
        if num_cells_4 > 0:
            if bidirectional: net = Bidirectional(CuDNNLSTM(num_cells_4, return_sequences=True))(net)
            else:             net = CuDNNLSTM(num_cells_4, return_sequences=True)(net)
            if batch_norm: net = BatchNormalization()(net)
            net = Activation('tanh')(net)
            net = Dropout(rate=dropout)(net)

        if comb_smoothing:  # TODO: does not really work yet, needs auxiliary loss, be careful when used with last specific
            net = CuDNNLSTM(num_targets, return_sequences=True)(net)
            net = Activation('tanh')(net)

        # outputs (& task-specific layers)
        out = []
        for n in range(num_targets):
            if last_lstm: outn = CuDNNLSTM(1, return_sequences=True)(net)
            else:         outn = TimeDistributed(Dense(1))(net)
            outn = Activation(final_activation)(outn)
            out.append(outn)
    else:  # 4th layer mandatory!
        out = []
        for n in range(num_targets):
            if bidirectional: net_part = Bidirectional(CuDNNLSTM(num_cells_4, return_sequences=True))(net)
            else:             net_part = CuDNNLSTM(num_cells_4, return_sequences=True)(net)
            if batch_norm: net_part = BatchNormalization()(net_part)
            net_part = Activation('tanh')(net_part)
            net_part = Dropout(rate=dropout)(net_part)
            #
            if last_lstm: outn = CuDNNLSTM(1, return_sequences=True)(net_part)
            else:         outn = TimeDistributed(Dense(1))(net_part)
            outn = Activation(final_activation)(outn)
            out.append(outn)

    return inputs, out


def generate_model_cnn(max_seq_len, num_features, num_targets, num_cells_1, num_cells_2, num_cells_3, num_cells_4, batch_norm, last_specific, final_activation, dropout=0.0):
    # Input
    inputs = Input(shape=(max_seq_len,num_features))
    net = inputs
    #
    # 1st layer
    if num_cells_1[0] > 0:
        net = Conv1D(num_cells_1[0], num_cells_1[1], strides=1, padding='same')(net)
        if batch_norm: net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = Dropout(rate=dropout)(net)
    #
    # 2nd layer
    if num_cells_2[0] > 0:
        net = Conv1D(num_cells_2[0], num_cells_2[1], strides=1, padding='same')(net)
        if batch_norm: net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = Dropout(rate=dropout)(net)
    #
    # 3rd layer
    if num_cells_3[0] > 0:
        net = Conv1D(num_cells_3[0], num_cells_3[1], strides=1, padding='same')(net)
        if batch_norm: net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = Dropout(rate=dropout)(net)
    #
    if not last_specific:
        # 4th layer
        if num_cells_4[0] > 0:
            net = Conv1D(num_cells_4[0], num_cells_4[1], strides=1, padding='same')(net)
            if batch_norm: net = BatchNormalization()(net)
            #net = Activation('relu')(net)
            # out = net
            # net = Dropout(rate=dropout)(net)
        #
        # outputs (& task-specific layers)
        out = []
        for n in range(num_targets):
            outn = TimeDistributed(Dense(1))(net)
            outn = Activation(final_activation)(outn)
            out.append(outn)

    else:  # 4th layer mandatory!
        out = []
        for n in range(num_targets):
            net_part = Conv1D(num_cells_4[0], num_cells_4[1], strides=1, padding='same')(net)
            if batch_norm: net_part = BatchNormalization()(net_part)
            net_part = Activation('relu')(net_part)
            net_part = Dropout(rate=dropout)(net_part)
            #
            outn = TimeDistributed(Dense(1))(net_part)
            outn = Activation(final_activation)(outn)
            out.append(outn)
    #
    return inputs, out


def evaluate_all_cross(model, trainX, develX, testX, crossX, trainY, develY, testY, crossY, origTrain, origDevel, origTest, origCross, shift, factor, num_targets):
    CCC_train          = np.zeros(num_targets)
    CCC_devel          = np.zeros(num_targets)
    CCC_test           = np.zeros(num_targets)
    CCC_testC          = np.zeros(num_targets)
    CCC_orig_train     = np.zeros(num_targets)
    CCC_orig_devel     = np.zeros(num_targets)
    CCC_orig_test      = np.zeros(num_targets)
    CCC_orig_testC     = np.zeros(num_targets)
    CCC_orig_train_seq = np.zeros(num_targets)
    CCC_orig_devel_seq = np.zeros(num_targets)
    CCC_orig_test_seq  = np.zeros(num_targets)
    CCC_orig_test_seqC = np.zeros(num_targets)
    CCC_orig_devel_pp  = np.zeros(num_targets)
    CCC_orig_test_pp   = np.zeros(num_targets)
    CCC_orig_test_ppC  = np.zeros(num_targets)

    # Get predictions
    predYtrain = model.predict(trainX)
    predYdevel = model.predict(develX)
    predYtest  = model.predict(testX)
    predYcross = model.predict(crossX)

    if num_targets==1:  # In this case, model.predict() does not return a list, which would be required
        predYtrain = [predYtrain]
        predYdevel = [predYdevel]
        predYtest  = [predYtest]
        predYcross = [predYcross]

    # Eval with the upsampled sampling rate and padded sequences
    for k in range(0,num_targets):  # loop over target dimensions (arousal, valence, liking)
        CCC_train[k] = compute_ccc(predYtrain[k].flatten(), trainY[k].flatten())
        CCC_devel[k] = compute_ccc(predYdevel[k].flatten(), develY[k].flatten())
        CCC_test[k]  = compute_ccc(predYtest[k].flatten(),  testY[k].flatten())
        CCC_testC[k] = compute_ccc(predYcross[k].flatten(), crossY[k].flatten())

    # Eval with downsampling - with the original labels
    # First shift predictions back (delay)
    for k in range(0,num_targets):
        predYtrain[k] = shift_annotations_to_back(predYtrain[k], shift)
        predYdevel[k] = shift_annotations_to_back(predYdevel[k], shift)
        predYtest[k]  = shift_annotations_to_back(predYtest[k], shift)
        predYcross[k] = shift_annotations_to_back(predYcross[k], shift)

    num_samples_down = int(np.round(float(trainY[0].shape[1])/factor))

    for k in range(0,num_targets):
        CCC_orig_train[k], CCC_orig_train_seq[k], _              , _               = evaluate_partition(predYtrain[k], num_samples_down, origTrain[k])
        CCC_orig_devel[k], CCC_orig_devel_seq[k], pred_list_devel, orig_list_devel = evaluate_partition(predYdevel[k], num_samples_down, origDevel[k])
        CCC_orig_test[k],  CCC_orig_test_seq[k],  pred_list_test,  orig_list_test  = evaluate_partition(predYtest[k],  num_samples_down, origTest[k])
        CCC_orig_testC[k], CCC_orig_test_seqC[k], pred_list_testC, orig_list_testC = evaluate_partition(predYcross[k], num_samples_down, origCross[k])

        # With postprocessed labels
        CCC_pp_devel, best_param = postprocess_labels.train(orig_list_devel,   pred_list_devel)
        CCC_pp_test              = postprocess_labels.predict(orig_list_test,  pred_list_test, best_param)
        CCC_pp_testC             = postprocess_labels.predict(orig_list_testC, pred_list_testC, best_param)

        CCC_orig_devel_pp[k] = CCC_pp_devel[-1]
        CCC_orig_test_pp[k]  = CCC_pp_test[-1]
        CCC_orig_test_ppC[k] = CCC_pp_testC[-1]
        #print(CCC_pp_devel)
        #print(CCC_pp_test)

    return CCC_train, CCC_devel, CCC_test, CCC_testC, CCC_orig_train, CCC_orig_devel, CCC_orig_test, CCC_orig_testC, CCC_orig_train_seq, CCC_orig_devel_seq, CCC_orig_test_seq, CCC_orig_test_seqC, CCC_orig_devel_pp, CCC_orig_test_pp, CCC_orig_test_ppC


def evaluate_all(model, trainX, develX, testX, trainY, develY, testY, origTrain, origDevel, origTest, shift, factor, num_targets):
    CCC_train          = np.zeros(num_targets)
    CCC_devel          = np.zeros(num_targets)
    CCC_test           = np.zeros(num_targets)
    CCC_orig_train     = np.zeros(num_targets)
    CCC_orig_devel     = np.zeros(num_targets)
    CCC_orig_test      = np.zeros(num_targets)
    CCC_orig_train_seq = np.zeros(num_targets)
    CCC_orig_devel_seq = np.zeros(num_targets)
    CCC_orig_test_seq  = np.zeros(num_targets)
    CCC_orig_devel_pp  = np.zeros(num_targets)
    CCC_orig_test_pp   = np.zeros(num_targets)

    # Get predictions
    predYtrain = model.predict(trainX)
    predYdevel = model.predict(develX)
    predYtest  = model.predict(testX)

    if num_targets==1:  # In this case, model.predict() does not return a list, which would be required
        predYtrain = [predYtrain]
        predYdevel = [predYdevel]
        predYtest  = [predYtest]

    # Eval with the upsampled sampling rate and padded sequences
    for k in range(0,num_targets):  # loop over target dimensions (arousal, valence, liking)
        CCC_train[k] = compute_ccc(predYtrain[k].flatten(), trainY[k].flatten())
        CCC_devel[k] = compute_ccc(predYdevel[k].flatten(), develY[k].flatten())
        CCC_test[k]  = compute_ccc(predYtest[k].flatten(),  testY[k].flatten())

    # Eval with downsampling - with the original labels
    # First shift predictions back (delay)
    for k in range(0,num_targets):
        predYtrain[k] = shift_annotations_to_back(predYtrain[k], shift)
        predYdevel[k] = shift_annotations_to_back(predYdevel[k], shift)
        predYtest[k]  = shift_annotations_to_back(predYtest[k], shift)

    num_samples_down = int(np.round(float(trainY[0].shape[1])/factor))

    for k in range(0,num_targets):
        CCC_orig_train[k], CCC_orig_train_seq[k], _              , _               = evaluate_partition(predYtrain[k], num_samples_down, origTrain[k])
        CCC_orig_devel[k], CCC_orig_devel_seq[k], pred_list_devel, orig_list_devel = evaluate_partition(predYdevel[k], num_samples_down, origDevel[k])
        CCC_orig_test[k],  CCC_orig_test_seq[k],  pred_list_test,  orig_list_test  = evaluate_partition(predYtest[k],  num_samples_down, origTest[k])

        # With postprocessed labels
        CCC_pp_devel, best_param = postprocess_labels.train(orig_list_devel,  pred_list_devel)
        CCC_pp_test              = postprocess_labels.predict(orig_list_test, pred_list_test, best_param)

        CCC_orig_devel_pp[k] = CCC_pp_devel[-1]
        CCC_orig_test_pp[k]  = CCC_pp_test[-1]
        #print(CCC_pp_devel)
        #print(CCC_pp_test)

    return CCC_train, CCC_devel, CCC_test, CCC_orig_train, CCC_orig_devel, CCC_orig_test, CCC_orig_train_seq, CCC_orig_devel_seq, CCC_orig_test_seq, CCC_orig_devel_pp, CCC_orig_test_pp


def evaluate_partition(pred, num_samples_down, orig):
    # pred (num_instances, max_seq_len, 1)
    # Evaluate with original labels in two ways
    # num_samples_down: constant length after downsampling (given constant length of input)
    CCC_orig     = np.zeros(pred.shape[2])
    CCC_orig_seq = np.zeros(pred.shape[2])
    pred_list    = []  # required by label postprocessing script
    orig_list    = []  # required by label postprocessing script - only one target

    CCC_seq = np.array([])
    predAll = np.array([])
    origAll = np.array([])
    for m in range(0,pred.shape[0]):
        # resampling
        predDown = scipy.signal.resample(pred[m,:],num_samples_down)
        # cropping or padding
        lenOrig = len(orig[m])
        if len(predDown)>lenOrig:
            predDown = predDown[:lenOrig]
        elif len(predDown)<lenOrig:
            predDown = np.concatenate((predDown,np.zeros(lenOrig-len(predDown))))
        # segment avg eval
        CCC      = compute_ccc(predDown.flatten(),orig[m].flatten())
        CCC_seq  = np.append(CCC_seq,CCC)
        # global eval
        predAll  = np.append(predAll,predDown)
        origAll  = np.append(origAll,orig[m])
        # append to lists
        pred_list.append(predDown.flatten())
        orig_list.append(orig[m].flatten())

    CCC_orig     = compute_ccc(predAll,origAll)  # global
    CCC_orig_seq = np.mean(CCC_seq)  # segment average
    return CCC_orig, CCC_orig_seq, pred_list, orig_list


def shift_annotations_to_front(labels, shift=0):
    labels = np.concatenate((labels[:,shift:,:], np.zeros((labels.shape[0],shift,labels.shape[2]))), axis=1)
    return labels


def shift_annotations_to_back(labels, shift=0):
    labels = np.concatenate((np.zeros((labels.shape[0],shift,labels.shape[2])), labels[:,:labels.shape[1]-shift,:]), axis=1)
    return labels


def write_csv(filename,matrix):
    with open(filename,'w') as file:
        for k in range(0,len(matrix)):
            for m in range(0,matrix.shape[1]):
                file.write(str(matrix[k,m]))
                if m<matrix.shape[1]-1:
                    file.write(',')
                else:
                    file.write('\n')

def append_csv(filename,line):
    with open(filename,'a') as file:
        for m in range(0,len(line)):
            file.write(str(line[m]))
            if m<len(line)-1:
                file.write(',')
            else:
                file.write('\n')


if __name__ == '__main__':
    main()

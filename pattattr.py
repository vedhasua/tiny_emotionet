from keras.models import Model, load_model, model_from_json
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

def main():
    # Configuration
    if True:
        modelFile          = '/home/vedhas/workspace/is2019_recurrence/results/5_oV_iv/m242_tr0.905_dv0.582_ts0.589_tc0.393.h5'
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
        max_num_epochs   = 500       # 
        first_lstm       = False    # only for LSTM models
        num_cells_1      = [200,5]   # if scalar: LSTM model
        num_cells_2      = [64,20]  # CNN model requires list [num_filters, filter_length]: =[0,0] means no 2nd layer
        num_cells_3      = [32,30]    # CNN model requires list [num_filters, filter_length]: =[0,0] means no 2nd layer
        num_cells_4      = [32,50]    # CNN model requires list [num_filters, filter_length]: =[0,0] means no 2nd layer
        last_lstm        = False    # only for LSTM models
        batch_norm       = False    # 
        last_specific    = False    # 
        comb_smoothing   = False    # only for LSTM models
        bidirectional    = False    # only for LSTM models
        dropout          = 0.0      # dropout - same for all
        final_activation = 'linear' # 'linear' or 'tanh'
        loss_function    = 'ccc_2'  # 
        
        shift_sec        = 2.8      #PARAM - 0.05 opt for window size 0.1, uni-directional LSTM
        targets_avl      = 'V'      # 
        feature_type_a   = 'funcegem'   # see sewa_data.py
        feature_type_v   = 'faus'   # see sewa_data.py
        window_size      = 0.1      # only relevant for egem, boaw, func
        xbow_cs          = 1000      # Codebook size, only relevant for boaw
        xbow_na          = 10        # Number of assignments, only relevant for boaw
        random_seed      = 0        # 
        add_noise        = False    # TODO: Noise layer (not yet implemented)

        fsuffix           = ""
        if 'audio' in modality:
            fsuffix += "_a_{}".format(feature_type_a)
        if 'video' in modality:
            fsuffix += "_v_{}".format(feature_type_v)

        DeNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/De{}.npy'.format(fsuffix)
        HuNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/Hu{}.npy'.format(fsuffix)
    
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
        trainX, trainY, develX, develY, testX, testY, origTrain, origDevel, origTest = np.load(DeNPY)
    else:
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
    ## Cross
    if eval_cross:
        if os.path.isfile(HuNPY): 
            trainHX, trainHY, develHX, develHY, testHX, testHY, origHTrain, origHDevel, origHTest, origCross, crossX, crossY = np.load(HuNPY)
        else:
            trainHX, trainHY, develHX, develHY, testHX, testHY, origHTrain, origHDevel, origHTest = load_SEWA(culture='Hungarian', 
                                                                                                          modality=modality, 
                                                                                                          mode_audio=feature_type_a, 
                                                                                                          mode_video=feature_type_v, 
                                                                                                          get_turn_feature=get_turn_feature, 
                                                                                                          window_size=window_size, 
                                                                                                          xbow_cs=xbow_cs, 
                                                                                                          xbow_na=xbow_na, 
                                                                                                          targets=targets)  # get also turn annotations
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

    ## Cross    
    max_seq_len  = trainX.shape[1]  # same for all
    num_features = trainX.shape[2]  # same for all X
    num_targets  = len(targets)
    
    print('Shifting labels to the front for ' + str(shift_sec) + ' seconds.')
    for t in range(0, num_targets):
        OrTy0, OrDv0, OrTs0 = [trainY[t], develY[t], testY[t]]
        trainY[t] = shift_annotations_to_front(trainY[t], shift)
        develY[t] = shift_annotations_to_front(develY[t], shift)
        testY[t]  = shift_annotations_to_front(testY[t],  shift)
        if eval_cross:
            crossY[t]  = shift_annotations_to_front(crossY[t],  shift)
        OrTy1, OrDv1, OrTs1 = [shift_annotations_to_back(trainY[t], shift),
                                shift_annotations_to_back(develY[t], shift),
                                shift_annotations_to_back(testY[t], shift)]
    fig=plt.figure()
    plt.plot(np.concatenate(np.concatenate(develY[t],axis=0),axis=0))
    #plt.plot(np.concatenate(np.concatenate(OrTy1,axis=0),axis=0))
    #plt.plot(np.concatenate(np.concatenate(TrainX,axis=0),axis=0))
    #plt.plot(OrDv)
    #plt.plot(OrTs)    
    # create and compile model
    # keras.losses.custom_loss = ccc_loss_2    
    model = load_model(modelFile, custom_objects={'ccc_loss_2': ccc_loss_2})
    # model = model_from_json(open(modelFile).read())
    # model.load_weights(os.path.join(os.path.dirname(modelFile), 'model_weights.h5'))
    print(model.summary())
    
    CCCs          = np.empty((0,num_targets*3))
    CCCs_orig     = np.empty((0,num_targets*3))
    CCCs_orig_seq = np.empty((0,num_targets*3))
    CCCs_orig_pp  = np.empty((0,num_targets*2))
    if eval_cross:     
        CCCs          = np.empty((0,num_targets*4))
        CCCs_orig     = np.empty((0,num_targets*4))
        CCCs_orig_seq = np.empty((0,num_targets*4))
        CCCs_orig_pp  = np.empty((0,num_targets*3))
    cccmax=0
    epoch = 1
    if True:
        print("Iter: " + str(epoch))
        
        # Evaluate and save results (in terms of CCC)
        if eval_cross:
            CCC_train, CCC_devel, CCC_test, CCC_testC, CCC_orig_train, CCC_orig_devel, CCC_orig_test, CCC_orig_testC, CCC_orig_train_seq, CCC_orig_devel_seq, CCC_orig_test_seq, CCC_orig_test_seqC, CCC_orig_devel_pp, CCC_orig_test_pp, CCC_orig_test_ppC = evaluate_all_cross(model, trainX, develX, testX, crossX, trainY, develY, testY, crossY, origTrain, origDevel, origTest, origCross, shift, factor_tr_orig, num_targets,plt)
            CCCs          = np.append(CCCs,         [np.array([CCC_train,          CCC_devel,          CCC_test,          CCC_testC]).flatten()],          axis=0)
            CCCs_orig     = np.append(CCCs_orig,    [np.array([CCC_orig_train,     CCC_orig_devel,     CCC_orig_test,     CCC_orig_testC]).flatten()],     axis=0)
            CCCs_orig_seq = np.append(CCCs_orig_seq,[np.array([CCC_orig_train_seq, CCC_orig_devel_seq, CCC_orig_test_seq, CCC_orig_test_seqC]).flatten()], axis=0)
            CCCs_orig_pp  = np.append(CCCs_orig_pp, [np.array([CCC_orig_devel_pp,  CCC_orig_test_pp,   CCC_orig_test_ppC                    ]).flatten()], axis=0)
            print("CCC train: {:.3f},{:.3f},{:.3f}, ".format(CCC_train[0],CCC_orig_train[0],CCC_orig_train_seq[0]) + \
                  "CCC devel: {:.3f},{:.3f},{:.3f},{:.3f} ".format(CCC_devel[0],CCC_orig_devel[0],CCC_orig_devel_seq[0],CCC_orig_devel_pp[0]) + \
                  "CCC test:  {:.3f},{:.3f},{:.3f},{:.3f} ".format(CCC_test[0], CCC_orig_test[0], CCC_orig_test_seq[0], CCC_orig_test_pp[0] ) + \
                  "CCC testC: {:.3f},{:.3f},{:.3f},{:.3f}".format(CCC_testC[0],CCC_orig_testC[0],CCC_orig_test_seqC[0],CCC_orig_test_ppC[0]))

    print("Done")


def evaluate_all_cross(model, trainX, develX, testX, crossX, trainY, develY, testY, crossY, origTrain, origDevel, origTest, origCross, shift, factor, num_targets, plt=None):
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
    # plt.plot(predYtest[k].flatten())
    # plt.plot(testY[k].flatten())
    # Eval with downsampling - with the original labels
    # First shift predictions back (delay)
    for k in range(0,num_targets):
        #plt.plot(predYtrain[k].flatten(),'r')
        #plt.plot(predYdevel[k].flatten(),'r')
        #plt.plot(predYtest[k].flatten(),'g')
        predYtrain[k] = shift_annotations_to_back(predYtrain[k], shift)
        predYdevel[k] = shift_annotations_to_back(predYdevel[k], shift)
        predYtest[k]  = shift_annotations_to_back(predYtest[k], shift)
        predYcross[k] = shift_annotations_to_back(predYcross[k], shift)
    
    if plt:    
        #plt.plot(predYtrain[k].flatten(),'g')
        plt.plot(predYdevel[k].flatten(),'g')
        #plt.plot(predYtest[k].flatten(),'g')
    print(compute_ccc(predYtest[k].flatten(),  testY[k].flatten()))
    num_samples_down = int(np.round(float(trainY[0].shape[1])/factor))
    
    for k in range(0,num_targets):
        CCC_orig_train[k], CCC_orig_train_seq[k], _              , _               = evaluate_partition2(predYtrain[k], origTrain[k])
        CCC_orig_devel[k], CCC_orig_devel_seq[k], pred_list_devel, orig_list_devel = evaluate_partition2(predYdevel[k], origDevel[k])
        CCC_orig_test[k],  CCC_orig_test_seq[k],  pred_list_test,  orig_list_test  = evaluate_partition2(predYtest[k], origTest[k],plt)
        CCC_orig_testC[k], CCC_orig_test_seqC[k], pred_list_testC, orig_list_testC = evaluate_partition2(predYcross[k],  origCross[k])
        plt.show()        

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


def evaluate_partition2(pred, orig, plt=None):
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
        predDown = pred[m,:]
        lenOrig = len(orig[m])
        if len(predDown)>lenOrig:
            predDown = predDown[:lenOrig]
        elif len(predDown)<lenOrig:
            predDown = np.concatenate((predDown.flatten(),np.zeros(lenOrig-len(predDown))))
        CCC      = compute_ccc(predDown,orig[m].flatten())
        CCC_seq  = np.append(CCC_seq,CCC)
        predAll  = np.append(predAll,predDown)
        origAll  = np.append(origAll,orig[m])
        # append to lists
        pred_list.append(predDown.flatten())
        orig_list.append(orig[m].flatten())
    CCC_orig     = compute_ccc(predAll,origAll)  # global
    CCC_orig_seq = np.mean(CCC_seq)  # segment average
    return CCC_orig, CCC_orig_seq, pred_list, orig_list



def evaluate_partition(pred, num_samples_down, orig, plt=None):
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
        if plt:
            plt.figure()
            plt.plot(pred[m,:])
            plt.plot(predDown)
            plt.plot(orig[m])
            plt.show()
        # global eval
        predAll  = np.append(predAll,predDown)
        origAll  = np.append(origAll,orig[m])
        # append to lists
        pred_list.append(predDown.flatten())
        orig_list.append(orig[m].flatten())
    #if fig:
    #    fig.plot(predAll)
    #    fig.plot(origAll)
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


import numpy as np
import scipy.signal
from standardise import standardise_3

# Helper functions
def get_num_lines(filename, skip_header=False):
    with open(filename, 'r') as file:
        c = 0
        if skip_header:
            c = -1
        for line in file:
            c += 1
    return c

def get_num_columns(filename, delim=';', skip_header=False):
    with open(filename, 'r') as file:
        if skip_header:
            next(file)
        line = next(file)
        offset1 = line.find(delim)+1
        offset2 = line[offset1:].find(delim)+1+offset1
        cols = np.fromstring(line[offset2:], dtype=float, sep=delim)
    return len(cols)

def read_csv(filename, num_lines=0, delim=';', skip_header=False):
    if num_lines==0:
        num_lines = get_num_lines(filename, skip_header)
    data = np.empty((num_lines,get_num_columns(filename,delim,skip_header)), float)
    with open(filename, 'r') as file:
        if skip_header:
            next(file)
        c = 0
        for line in file:
            offset1 = line.find(delim)+1
            offset2 = line[offset1:].find(delim)+1+offset1
            data[c,:] = np.fromstring(line[offset2:], dtype=float, sep=delim)
            c += 1
    return data


def load_annotations(culture, part, num_inst, targets=[0,1,2], max_seq_len_labels=1768, upsampling_factor=1, invert_std=False):
    # upsampling_factor=10 required for raw audio features representation
    path_package      = culture
    max_seq_len_final = max_seq_len_labels * upsampling_factor
    
    if culture=='German':
        cabr = 'DE'
        num_annos = 6
    elif culture=='Hungarian':
        cabr = 'HU'
        num_annos = 5
    
    annos_original      = []
    annos_continued     = []
    #annos_numpy        = []  # TODO: not sure if this is useful for something
    annos_original_std  = []
    annos_continued_std = []
    
    for t in targets:
        min_index = t*num_annos + 3
        max_index = min_index + num_annos
        
        # Single annotations
        annos_original_t  = []  # Original (for evaluation)
        annos_continued_t = []  # Continued (for training)
        for a in range(0,num_annos):
            annotator_continued_t = np.zeros((num_inst, max_seq_len_final, 1))
            annotator_original_t  = []
            for n in range(0,num_inst):
                yn = read_csv(path_package + '/labels_annos/std/' + part + '_' + cabr + '_' + str(n+1).zfill(2) + '.csv')
                yn = yn[:,min_index+a].reshape((yn.shape[0], 1))  # select only target dimension and reshape to 2D array
                # original
                annotator_original_t.append(yn)
                # continued
                seq_len_after_res = yn.shape[0]*upsampling_factor
                yn_resampled = scipy.signal.resample(yn, seq_len_after_res)  # resampling (only if required)
                while yn_resampled.shape[0] < max_seq_len_final:
                    yn_resampled = np.concatenate((yn_resampled,yn_resampled),axis=0)  # continuation ...
                annotator_continued_t[n,:,:] = yn_resampled[:max_seq_len_final,:]      # & cropping
            annos_original_t.append(annotator_original_t)
            annos_continued_t.append(annotator_continued_t)
        annos_original.append(annos_original_t)
        annos_continued.append(annos_continued_t)
        
        ## Numpy
        #annos_numpy_t = []
        #for n in range(0,num_inst):
        #    yn = read_csv(path_package + '/labels_annos/std/' + part + '_' + cabr + '_' + str(n+1).zfill(2) + '.csv')
        #    yn = yn[:,min_index:max_index]
        #    yn = np.transpose(yn, [1,0])
        #    yn = yn.reshape((num_annos, -1, 1))
        #    annos_numpy_t.append(yn)
        #annos_numpy.append(annos_numpy_t)
        
        ## Standard deviation of the annotator contours
        annos_original_std_t  = []
        annos_continued_std_t = np.zeros((num_inst, max_seq_len_final, 1))
        for n in range(0,num_inst):
            yn = read_csv(path_package + '/labels_annos/std/' + part + '_' + cabr + '_' + str(n+1).zfill(2) + '.csv')
            yn = yn[:,min_index:max_index]
            yn = np.std(yn, axis=1)  # Standard deviation
            if invert_std:
                yn = 1. - yn
            yn = yn.reshape((-1, 1))
            # original
            annos_original_std_t.append(yn)
            # Padded (and upsampled)
            seq_len_after_res = yn.shape[0]*upsampling_factor
            yn_resampled = scipy.signal.resample(yn, seq_len_after_res)  # resampling (if required)
            while yn_resampled.shape[0] < max_seq_len_final:
                yn_resampled = np.concatenate((yn_resampled,yn_resampled),axis=0)  # continuation ...
            annos_continued_std_t[n,:,:] = yn_resampled[:max_seq_len_final,:]      # & cropping
        annos_original_std.append(annos_original_std_t)
        annos_continued_std.append(annos_continued_std_t)
    
    return annos_original, annos_continued, annos_original_std, annos_continued_std


def load_labels(culture, part, num_inst, max_seq_len_labels, upsampling_factor=1, targets=[0,1,2]):
    # upsampling_factor=10 required for raw audio features representation
    path_package      = culture
    max_seq_len_final = max_seq_len_labels * upsampling_factor
    
    labels_original  = []
    labels_continued = []
    
    for t in targets:
        labels_original_t  = []
        labels_continued_t = np.empty((num_inst, max_seq_len_final, 1))
        
        for n in range(0,num_inst):
            yn = read_csv(path_package + '/labels/' + part + '_' + str(n+1).zfill(2) + '.csv')
            yn = yn[:,t].reshape((yn.shape[0], 1))  # select only target dimension and reshape to 2D array
            # Original length labels
            labels_original_t.append(yn)
            # Continued (and upsampled)
            seq_len_after_res = yn.shape[0]*upsampling_factor
            yn_resampled = scipy.signal.resample(yn, seq_len_after_res)  # resampling (if required)
            while yn_resampled.shape[0] < max_seq_len_final:
                yn_resampled = np.concatenate((yn_resampled,yn_resampled),axis=0)  # continuation ...
            labels_continued_t[n,:,:] = yn_resampled[:max_seq_len_final,:]         # & cropping
        labels_original.append(labels_original_t)
        labels_continued.append(labels_continued_t)
    

    return labels_original, labels_continued


def compute_functionals(X, max_seq_len_out, hop_size, window_size):
    # hop_size & window_size in #samples / timesteps in X
    
    X_func = np.zeros( (max_seq_len_out, X.shape[1]*2) )
    window_size_half = int(window_size/2)
    
    for t in range(0, max_seq_len_out):
        t_orig   = t * hop_size
        min_orig = max(0, t_orig-window_size_half)
        max_orig = min(X.shape[0], t_orig+window_size_half)
        if min_orig>=max_orig:
            break  # X might be smaller
        X_func[t,:X.shape[1]] = np.mean(X[min_orig:max_orig,:], axis=0)
        X_func[t,X.shape[1]:] = np.std(X[min_orig:max_orig,:],  axis=0)
    
    return X_func


def load_partition_audio(culture, part, num_inst, max_seq_len, get_turn_feature, mode='raw', window_size=1.0, xbow_cs=0, xbow_na=0):
    # mode: 'egem-raw':  eGeMAPS LLDs at original hop size of 0.01s, upsample labels to 0.01s
    #       'egem:       Original eGeMAPS functionals at a hop size of 0.1s (aligned with the labels)
    #       'egem-boaw': Boaw of LLDs with a hop size of 0.1s (aligned with the labels)
    #       'funcegem':  Functionals (mean and stddev) of LLDs with a hop size of 0.1s (aligned with the labels) and given window_size of  23 eGeMAPS LLDs
    #       'funccomp':  Functionals (mean and stddev) of LLDs with a hop size of 0.1s (aligned with the labels) and given window_size of 130 ComParE LLDs
    #       'mfcccomp':  Functionals (mean and stddev) of LLDs with a hop size of 0.1s (aligned with the labels) and given window_size of MFCCs (and deltas) from ComParE feature set
    #       'mfcc':      Functionals (mean and stddev) of LLDs with a hop size of 0.1s (aligned with the labels) and given window_size of  MFCCs (deltas and acceleration) from AVEC2018 challenge
    # xbow_param: only relevant for mode='boaw'; e.g.: xbow_param='100_1'
    
    path_package = culture
    
    if mode=='egem-raw':
        path_features     = 'audio_features'
        path_turns        = 'turn_features_audio'
        num_features      = 23  # turn feature does not count
        max_seq_len_final = max_seq_len
    else:
        path_turns        = 'turn_features_chunk'
        max_seq_len_final = int(np.ceil(max_seq_len/10.))  # seq len for reading features (with offset)
        
        if mode=='egem':
            path_features = 'audio_features_functionals_' + str(window_size) + 's'
            num_features  = 88  # turn feature does not count
        elif mode=='egem-boaw':
            path_features = 'audio_features_xbow_' + str(xbow_cs) + '_' + str(xbow_na) + '_' + str(window_size) + 's'
            num_features  = xbow_cs  # codebook size - TODO: split codebook not yet possible
        elif mode=='funcegem':
            path_features = 'audio_features'
            num_features  = 23*2  # mean & stddev for each LLD
            hop_size      = 10    # 10 frames = 100ms = 0.1s
            window_size   = int(np.round(window_size*100))  # w.r.t. original 100fps; 1.0 -> 100
        elif mode=='funccomp':
            path_features = 'audio_features_compare'
            num_features  = 130*2  # mean & stddev for each LLD
            hop_size      = 10     # 10 frames = 100ms = 0.1s
            window_size   = int(np.round(window_size*100))  # w.r.t. original 100fps; 1.0 -> 100
        elif mode=='mfcccomp':
            ind_mfcc      = np.array([53,54,55,56,57,58,59,60,61,62,63,64,65,66,118,119,120,121,122,123,124,125,126,127,128,129,130,131]) - 2  # -2 for name,timestamp are not present in the features array
            path_features = 'audio_features_compare'
            num_features  = len(ind_mfcc)*2   # mean & stddev for each LLD
            hop_size      = 10                # 10 frames = 100ms = 0.1s
            window_size   = int(np.round(window_size*100))  # w.r.t. original 100fps; 1.0 -> 100
        elif mode=='mfcc':
            path_features = 'audio_features_avec'
            num_features  = 39*2  # mean & stddev for each LLD
            hop_size      = 10    # 10 frames = 100ms = 0.1s
            window_size   = int(np.round(window_size*100))  # w.r.t. original 100fps; 1.0 -> 100
    
    if get_turn_feature: X = np.empty((num_inst,max_seq_len_final,num_features+1))
    else:                X = np.empty((num_inst,max_seq_len_final,num_features))
    
    for n in range(0,num_inst):
        # Features
        if mode=='mfcc':
            skip_header = True  # AVEC 2018 feature files
        else:
            skip_header = False
        features = read_csv(path_package + '/' + path_features + '/' + part + '_' + str(n+1).zfill(2) + '.csv', num_lines=0, delim=';', skip_header=skip_header)
        if mode=='mfcccomp':
            features = features[:,ind_mfcc]
        if mode=='funcegem' or mode=='funccomp' or mode=='mfcccomp' or mode=='mfcc':
            features = compute_functionals(features, max_seq_len_final, hop_size, window_size)
        
        if get_turn_feature:
            turn_feature = read_csv(path_package + '/' + path_turns + '/' + part + '_' + str(n+1).zfill(2) + '.csv').reshape((max_seq_len_final))  # should be padded already
            turn_feature = turn_feature[:features.shape[0]]  # was zero padded already -> remove this and continue (below)
        while features.shape[0]<max_seq_len_final:
            features = np.concatenate((features,features), axis=0)  # continuation ...
            if get_turn_feature: turn_feature = np.concatenate((turn_feature,turn_feature), axis=0)  # continuation ...
        X[n,:,:num_features] = features[:max_seq_len_final,:]  # & cropping
        if get_turn_feature: X[n,:,-1] = turn_feature[:max_seq_len_final]  # & cropping
    return X


def load_partition_video(culture, part, num_inst, max_seq_len, get_turn_feature, mode='raw', window_size=1.0):
    # mode:  'raw':   FAU confidences at original hop size of 0.02s, upsampling labels to 0.01s
    #        'func':  Functionals (mean and stddev) of FAUs with a hop size of 0.1s (aligned with the labels)
    
    path_package = culture
    
    if mode=='raw':
        path_features     = 'visual_features'
        path_turns        = 'turn_features_visual'
        num_features      = 18  # including confidence, turn feature does not count
        max_seq_len_final = max_seq_len
    else:
        path_turns        = 'turn_features_chunk'
        max_seq_len_final = int(np.ceil(max_seq_len/5.))  # seq len for reading features (with offset)
        
        if mode=='faus':
            path_features = 'visual_features'
            num_features  = 18*2  # mean & stddev for each LLD
            hop_size      = 5     # 5 video frames = 100ms = 0.1s
            window_size   = int(np.round(window_size*50))  # w.r.t. original 50fps; 1.0 -> 50
            skip_header   = True
        elif mode=='land':
            path_features = 'video_features_normalised'
            num_features  = 98*2 #121*2  # mean & stddev for each landmark (pose, landmarks)
            hop_size      = 5      # 5 video frames = 100ms = 0.1s
            window_size   = int(np.round(window_size*50))  # w.r.t. original 50fps; 1.0 -> 50
            skip_header   = False
        elif mode=='faus+lips':
            path_features = 'visual_features' # TODO (see below)
            num_features  = 18*2+12*2  #121*2  # mean & stddev for each landmark (pose, landmarks)
            hop_size      = 5     # 5 video frames = 100ms = 0.1s
            window_size   = int(np.round(window_size*50))  # w.r.t. original 50fps; 1.0 -> 50
            skip_header   = True # TODO (see below)
        else:
            print('Error: Visual feature mode not available: ' + mode)
    
    if get_turn_feature: X = np.empty((num_inst,max_seq_len_final,num_features+1))
    else:                X = np.empty((num_inst,max_seq_len_final,num_features))
    
    for n in range(0,num_inst):
        features = read_csv(path_package + '/' + path_features + '/' + part + '_' + str(n+1).zfill(2) + '.csv', num_lines=0, delim=';', skip_header=skip_header)
        if mode=='land':  # remove all the pose and eye features
            features = features[:,-98:]
        if mode=='faus+lips':  # remove all the pose and eye features
            lip_feat = read_csv(path_package + '/' + 'video_features_normalised' + '/' + part + '_' + str(n+1).zfill(2) + '.csv', num_lines=0, delim=';', skip_header=False)
            lip_featx = lip_feat[:len(features),-55:-49]
            lip_featy = lip_feat[:len(features),-6:]
            features = np.concatenate((features, lip_featx, lip_featy), axis=1)
        if mode!='raw':
            features = compute_functionals(features, max_seq_len_final, hop_size, window_size)
        
        if get_turn_feature:
            turn_feature = read_csv(path_package + '/' + path_turns + '/' + part + '_' + str(n+1).zfill(2) + '.csv').reshape((max_seq_len_final))  # should be padded already
            turn_feature = turn_feature[:features.shape[0]]  # was zero padded already -> remove this and continue (below)
        while features.shape[0]<max_seq_len_final:
            features     = np.concatenate((features,features), axis=0)  # continuation ...
            if get_turn_feature: turn_feature = np.concatenate((turn_feature,turn_feature), axis=0)  # continuation ...
        X[n,:,:num_features] = features[:max_seq_len_final,:]  # & cropping
        if get_turn_feature: X[n,:,-1] = turn_feature[:max_seq_len_final]  # & cropping
    return X


def load_partition_linguistic(culture, part, num_inst, max_seq_len, get_turn_feature, mode='word2vec'):
    path_package = culture
    
    if mode=='word2vec':
        path_features     = mode + '_features'
        path_turns        = 'turn_features_chunk'  # word2vec features are already sampled at 10 fps (=label fps)
        num_features      = 300
        max_seq_len_final = max_seq_len  # must be aligned with label frequency
    else:
        print('Error: Linguistic feature mode not available: ' + mode)
    
    if get_turn_feature: X = np.empty((num_inst,max_seq_len_final,num_features+1))
    else:                X = np.empty((num_inst,max_seq_len_final,num_features))
    
    for n in range(0,num_inst):
        features = read_csv(path_package + '/' + path_features + '/' + part + '_' + str(n+1).zfill(2) + '.csv', num_lines=0, delim=';', skip_header=False)
        
        if get_turn_feature:
            turn_feature = read_csv(path_package + '/' + path_turns + '/' + part + '_' + str(n+1).zfill(2) + '.csv').reshape((max_seq_len_final))  # should be padded already
            turn_feature = turn_feature[:features.shape[0]]  # was zero padded already -> remove this and continue (below)        
        while features.shape[0]<max_seq_len_final:
            features     = np.concatenate((features,features), axis=0)          # continuation ...
            if get_turn_feature: turn_feature = np.concatenate((turn_feature,turn_feature), axis=0)  # continuation ...        
        X[n,:,:num_features] = features[:max_seq_len_final,:]  # & cropping
        if get_turn_feature: X[n,:,-1] = turn_feature[:max_seq_len_final]  # & cropping
    return X


def load_SEWA(culture='German', modality='audio', mode_audio='mfcc', mode_video='faus', get_turn_feature=True, window_size=1.0, xbow_cs=0, xbow_na=0, targets=[0,1,2]):
    if culture=='German':
        num_train    = 34
        num_devel    = 14
        num_test     = 16
    elif culture=='Hungarian':
        num_train    = 34
        num_devel    = 14
        num_test     = 18
    elif culture=='Chinese':
        num_train    = 0
        num_devel    = 0
        num_test     = 70
    
    if modality=='audio+video':
        print('Make sure that features are requested on chunk level (label frequency), not raw features! Otherwise, an error will occur!')
    
    max_seq_len_audio = 17679 # German: 17579
    max_seq_len_video = 8839  # German: 8791
    max_seq_len_chunk = 1768  # 
    
    # Initialise numpy arrays
    trainX = np.empty((num_train, max_seq_len_chunk, 0))
    develX = np.empty((num_devel, max_seq_len_chunk, 0))
    testX  = np.empty((num_test,  max_seq_len_chunk, 0))
    
    if modality.find('audio')>=0:
        trainX = np.concatenate((trainX, load_partition_audio(culture, 'Train', num_train, max_seq_len_audio, get_turn_feature, mode_audio, window_size, xbow_cs, xbow_na)), axis=2)
        develX = np.concatenate((develX, load_partition_audio(culture, 'Devel', num_devel, max_seq_len_audio, get_turn_feature, mode_audio, window_size, xbow_cs, xbow_na)), axis=2)
        testX  = np.concatenate((testX,  load_partition_audio(culture, 'Test',  num_test,  max_seq_len_audio, get_turn_feature, mode_audio, window_size, xbow_cs, xbow_na)), axis=2)
        if mode_audio!='egem-boaw':
            trainX, develX, testX = standardise_3(trainX, develX, testX)  # on-line standardisation
        get_turn_feature = False  # Add Turn Feature maximum once
    if modality.find('video')>=0:
        trainX = np.concatenate((trainX, load_partition_video(culture, 'Train', num_train, max_seq_len_video, get_turn_feature, mode_video, window_size)), axis=2)
        develX = np.concatenate((develX, load_partition_video(culture, 'Devel', num_devel, max_seq_len_video, get_turn_feature, mode_video, window_size)), axis=2)
        testX  = np.concatenate((testX,  load_partition_video(culture, 'Test',  num_test,  max_seq_len_video, get_turn_feature, mode_video, window_size)), axis=2)
        get_turn_feature = False  # Add Turn Feature maximum once
    if modality.find('linguistic')>=0:
        trainX = np.concatenate((trainX, load_partition_linguistic(culture, 'Train', num_train, max_seq_len_chunk, get_turn_feature)), axis=2)
        develX = np.concatenate((develX, load_partition_linguistic(culture, 'Devel', num_devel, max_seq_len_chunk, get_turn_feature)), axis=2)
        testX  = np.concatenate((testX,  load_partition_linguistic(culture, 'Test',  num_test,  max_seq_len_chunk, get_turn_feature)), axis=2)
        get_turn_feature = False  # Add Turn Feature maximum once
    
    if modality.find('audio')>=0 and mode_audio=='raw':
        upsampling_factor = 10
    elif modality.find('video')>=0 and mode_video=='raw':
        upsampling_factor = 5
    else:
        upsampling_factor = 1
    
    labelsTrain, trainY = load_labels(culture, 'Train', num_train, max_seq_len_chunk, upsampling_factor, targets)
    labelsDevel, develY = load_labels(culture, 'Devel', num_devel, max_seq_len_chunk, upsampling_factor, targets)
    labelsTest,  testY  = load_labels(culture, 'Test',  num_test,  max_seq_len_chunk, upsampling_factor, targets)
    
    return trainX, trainY, develX, develY, testX, testY, labelsTrain, labelsDevel, labelsTest


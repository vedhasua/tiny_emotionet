import time
import os
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from ccc import ccc_loss_1, ccc_loss_2, ccc_loss_3
from keras.models import Model, load_model, model_from_json
import seaborn as sns
from matplotlib import pyplot as plt
from ccc import compute_ccc
import scipy.signal
import glob
from standardise import standardise_3, reverse_standardise_3, reverse_rescale_3, scale_3
print(K.learning_phase())

def shift_annotations_to_front(labels, shift=0):
    labels = np.concatenate((labels[:,shift:,:], np.zeros((labels.shape[0],shift,labels.shape[2]))), axis=1)
    return labels

def shift_annotations_to_back(labels, shift=0):
    labels = np.concatenate((np.zeros((labels.shape[0],shift,labels.shape[2])), labels[:,:labels.shape[1]-shift,:]), axis=1)
    return labels

origf,stdf,meanf=1,0,0
trystdf=0 and origf
columnNames = [
            #'confidence', 'success',
             'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz',                       
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r',    
        'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r',           'AU45_r',    
        'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c',    
        'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']

shiftinsamples = 0
modelFile      = '/home/vedhas/workspace/EmoPain/BioVid/PartA_ACII2017_Paper/s42b1024e200r24n41.h5' #s42b1024e200r24n41.h5' #s42b1024e200r24n41_good.h5' #s42b256e200r24n41.h5' #s42b1024e200r24n41.h5'#s42b1024e200r24.h5'
model = load_model(modelFile, custom_objects={'ccc_loss_2': ccc_loss_2})
model.layers[2].trainable=False
print([layer.name for layer in model.layers])
print(model.layers[2].get_weights())
#print(model.layers[2].params())
batch_norm_exists=False
if 'batch_normalization_1' in [layer.name for layer in model.layers]:
    filters, biases = model.layers[1].get_weights()
    gamma, beta, meanC, stdC = model.layers[2].get_weights()
    stdC=np.sqrt(stdC)+0.001
    gamma=gamma
    wdense, bdense = model.layers[3].get_weights()
    batch_norm_exists=True
else:
    filters, biases = model.layers[1].get_weights()
    wdense, bdense = model.layers[2].get_weights()

wdense = wdense.flatten()
bdense = bdense.flatten()
biases = biases.flatten()
print(filters.shape)
#print(filters, biases)
print(wdense, bdense)

fig=plt.figure(figsize=(8.5,7))
myheatmap=np.transpose(filters[:,:,0])*wdense
myheatmap=np.vstack((myheatmap,np.expand_dims(myheatmap.mean(axis=0),0)))
myheatmap=np.hstack((myheatmap,np.expand_dims(myheatmap.mean(axis=1),1)))
myheatmap[-1,-1]=0
g=sns.heatmap(myheatmap,cmap="PiYG",yticklabels=columnNames+['Mean W_t'],xticklabels=4,center=0.000,linewidths=0.5,linecolor='gray')
locs, labels = plt.xticks()
labels[-1] = 'Mean W_f'
plt.xticks(locs, labels)
g.set_yticklabels(g.get_yticklabels(), rotation = 0)
g.set_xticklabels(list(np.arange(1,4,25)), minor=True)
bottom, top = g.get_ylim()
g.set_ylim(bottom + 0.5, top - 0.5)

plt.tight_layout()
plt.show()
'''
###########
# NEW PLOT
SimpleOne=np.ones((1,filters.shape[0],filters.shape[1]))
myheatmap=np.multiply(SimpleOne,filters[:,:,0]))*wdense + (biases.flatten()*wdense+bdense)/(filters.shape[0]*filters.shape[1])
score=np.transpose(myheatmap)
score=score[:,:,0]
fig=plt.figure(figsize=(8.5,7))
g = sns.heatmap(score,cmap="PiYG",yticklabels=columnNames,xticklabels= 4,center=0.000,linewidths=0.5,linecolor='gray')
g.set_yticklabels(g.get_yticklabels(), rotation = 0)
g.set_xticklabels(list(np.arange(1,4,25)), minor=True)
bottom, top = g.get_ylim()
g.set_ylim(bottom + 0.5, top - 0.5)
plt.tight_layout()
plt.show()
########
'''

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


testtX_orig=testtX
apply_scaling = True
if apply_scaling:
    trainX, validX, testtX, MEAN_X, STDDEV_X = standardise_3(trainX, validX, testtX, return_mean_std=True)
    trainY, validY, testtY, MEAN_Y, STDDEV_Y = standardise_3(trainY, validY, testtY, return_mean_std=True)

from matplotlib import animation



X=testtX
Y=testtY
reverse_std = True
pY = model.predict(X)
pY  = shift_annotations_to_back(pY,  shiftinsamples)

if reverse_std:
    Y_rev  = reverse_standardise_3(Y,  STDDEV_Y, MEAN_Y)
    pY_rev = reverse_standardise_3(pY, STDDEV_Y, MEAN_Y)
    X_rev  = reverse_standardise_3(X,  STDDEV_X, MEAN_X)

CCC_orig     = np.zeros(pY.shape[2])
CCC_orig_seq = np.zeros(pY.shape[2])
pred_list    = []  # required by label postprocessing script
orig_list    = []  # required by label postprocessing script - only one target

CCC_seq = []
predAll = np.array([])
origAll = np.array([])

for subjectID in range(X.shape[0]):
    predDown=pY[subjectID,:]#scipy.signal.resample(pY[subjectID,:],num_samples_down)
    lenOrig = Y[subjectID].shape[0]
    assert lenOrig==138
    # segment avg eval
    CCC      = compute_ccc(predDown.flatten(),Y[subjectID].flatten())
    CCC_seq.append(CCC)
    # global eval
    predAll  = np.append(predAll,predDown)
    origAll  = np.append(origAll,Y[subjectID])
    # append to lists
    pred_list.append(predDown.flatten())
    orig_list.append(Y[subjectID].flatten())
CCC_orig     = compute_ccc(predAll,origAll)  # global
#print(CCC_seq, CCC_orig)

CCC_seq = []
predAll = np.array([])
origAll = np.array([])
for subjectID in range(X.shape[0]):
    predDown=pY_rev[subjectID,:]#scipy.signal.resample(pY[subjectID,:],num_samples_down)
    lenOrig = Y_rev[subjectID].shape[0]
    assert lenOrig==138
    if len(predDown)>lenOrig:
        predDown = predDown[:lenOrig]
    elif len(predDown)<lenOrig:
        predDown = np.concatenate((predDown,np.zeros(lenOrig-len(predDown))))
    # segment avg eval
    CCC      = compute_ccc(predDown.flatten(),Y_rev[subjectID].flatten())
    CCC_seq.append(CCC)
    # global eval
    predAll  = np.append(predAll,predDown)
    origAll  = np.append(origAll,Y_rev[subjectID])
    # append to lists
    pred_list.append(predDown.flatten())
    orig_list.append(Y_rev[subjectID].flatten())
CCC_orig     = compute_ccc(predAll,origAll)  # global
#print(CCC_seq, CCC_orig)

pause = False
def press(event):
    global pause
    pause = not pause

import matplotlib.patches as patches


fig1,(ax1,ax2)=plt.subplots(2,1,figsize=(8,7),gridspec_kw={'height_ratios': [4, 1]})
fig1.canvas.mpl_connect('key_press_event', press)

plt.tight_layout()
neighbourhood_by_2=30#int(1.5*filters.shape[0])
print(filters.shape)
print(biases.shape)
print(wdense.shape)
print(bdense.shape)
print("-------------")

num_filt_timesteps = filters.shape[0]
num_filt_features = filters.shape[1]
STDDEV_X_filtsize = np.tile(STDDEV_X+np.finfo(np.float32).eps,(num_filt_timesteps,1)).transpose()
MEAN_X_filtsize = np.tile(MEAN_X,(num_filt_timesteps,1)).transpose()
print(STDDEV_X_filtsize.shape,MEAN_X_filtsize.shape)
for subjectID in range(X.shape[0]):
    for t in range(26,138,2):
        if not pause and t-int(filters.shape[0]/2)>0 and t+int(filters.shape[0]/2)<138:
          try:
            #print(np.multiply(X[subjectID,t-int(filters.shape[0]/2):t+int(filters.shape[0]/2),:],filters[:,:,0]))
            ax1.clear()
            ax2.clear()
            #score=np.transpose(np.multiply(X[subjectID,t-int(filters.shape[0]/2):t+int(filters.shape[0]/2),:],filters[:,:,0]))*wdense + (biases.flatten()*wdense+bdense)/(filters.shape[0]*filters.shape[1])            
            use_scaled_scores=True

            if use_scaled_scores and batch_norm_exists and 0:
                print("I am using batchnormbased calculation")
                divideby = STDDEV_X_filtsize*stdC*num_filt_timesteps*num_filt_features
                curr_X = testtX_orig[subjectID,t-int(filters.shape[0]/2):t+int(filters.shape[0]/2),:]
                weight_filter = filters[:,:,0]
                inner11 = np.transpose(  np.multiply( num_filt_timesteps*num_filt_features*curr_X - MEAN_X_filtsize.transpose(), weight_filter )   )
                inner12 = biases.flatten()*STDDEV_X_filtsize
                inner13 = -meanC*STDDEV_X_filtsize
                inner21 = gamma*(inner11+inner12+inner13) + beta*stdC*STDDEV_X_filtsize
                del inner11, inner12, inner13
                inner31 = inner21*wdense + bdense*stdC*STDDEV_X_filtsize
                score = (STDDEV_Y*inner31 + MEAN_Y*stdC*STDDEV_X_filtsize)/divideby
            elif use_scaled_scores and batch_norm_exists and 1:
                print("I am using batchnormbased calculation2")
                divideby1 = STDDEV_X_filtsize*stdC
                divideby2 = num_filt_timesteps*num_filt_features
                weight_filter = filters[:,:,0]
                curr_X = testtX_orig[subjectID,t-int(filters.shape[0]/2):t+int(filters.shape[0]/2),:]
                core_score = np.transpose(  np.multiply(curr_X, weight_filter )   )
                core_score = core_score*gamma*wdense*STDDEV_Y
                offset_score = -1 * np.transpose(  np.multiply( MEAN_X_filtsize.transpose(),weight_filter )   ) 
                offset_score = offset_score*gamma*wdense*STDDEV_Y
                offset_score = offset_score + biases.flatten()*STDDEV_X_filtsize*gamma*wdense*STDDEV_Y
                offset_score = offset_score - meanC*STDDEV_X_filtsize*gamma*wdense*STDDEV_Y
                offset_score = offset_score + stdC*STDDEV_X_filtsize*beta*wdense*STDDEV_Y
                offset_score = offset_score + stdC*STDDEV_X_filtsize*bdense*STDDEV_Y
                offset_score = offset_score + stdC*STDDEV_X_filtsize*MEAN_Y
                core_score = core_score/divideby1
                offset_score = offset_score/(divideby1*divideby2)
                score2 = core_score+offset_score 
                #elif use_scaled_scores and batch_norm_exists and 0:
                print("I am using batchnormbased calculation3")
                divideby1 = STDDEV_X_filtsize*stdC
                divideby2 = num_filt_timesteps*num_filt_features
                weight_filter = filters[:,:,0]
                curr_X = testtX_orig[subjectID,t-int(filters.shape[0]/2):t+int(filters.shape[0]/2),:]
                core_score = np.transpose(  np.multiply(curr_X, weight_filter )   )
                core_score = core_score*gamma*wdense*STDDEV_Y/(STDDEV_X_filtsize*stdC)
                offset_score = -1 * np.transpose(  np.multiply( MEAN_X_filtsize.transpose(), weight_filter )   ) 
                offset_score = offset_score*gamma*wdense*STDDEV_Y/(STDDEV_X_filtsize*stdC)
                offset_score = offset_score + biases.flatten()*gamma*wdense*STDDEV_Y/(stdC)
                offset_score = offset_score - meanC*gamma*wdense*STDDEV_Y/(stdC)
                offset_score = offset_score + beta*wdense*STDDEV_Y
                offset_score = offset_score + bdense*STDDEV_Y
                offset_score = offset_score + MEAN_Y
                offset_score = offset_score/divideby2
                score = core_score+offset_score 
                print(np.sum(core_score), np.sum(offset_score), np.sum(score),pY_rev.flatten()[t])                
            elif use_scaled_scores and not batch_norm_exists and 1:
                #print("using scaled but non-batchnorm original score calculation")
                MEAN_X_filtsize=np.zeros_like(MEAN_X_filtsize)
                STDDEV_X_filtsize=np.ones_like(STDDEV_X_filtsize)
                #STDDEV_Y=1
                #MEAN_Y=0                
                core_score = np.transpose(np.multiply(testtX[subjectID,t-int(filters.shape[0]/2):t+int(filters.shape[0]/2),:],filters[:,:,0]))*wdense*STDDEV_Y/STDDEV_X_filtsize
                sc_meanX = -MEAN_X_filtsize*filters[:,:,0].transpose()*wdense*STDDEV_Y
                BfSxWdSy = biases*STDDEV_X_filtsize*wdense*STDDEV_Y
                BdSxSy = bdense*STDDEV_X_filtsize*STDDEV_Y
                MySx = MEAN_Y*STDDEV_X_filtsize
                offset_score = (sc_meanX + BfSxWdSy + BdSxSy + MySx)/(num_filt_timesteps*num_filt_features*STDDEV_X_filtsize) 
                #Wf_Wd_meanx = -MEAN_X_filtsize*filters[:,:,0].transpose()*wdense
                #Bf_Wd = biases.flatten()*wdense
                #offset_score = (
                #                Wf_Wd_meanx*STDDEV_Y +
                #         Bf_Wd*STDDEV_Y + bdense*STDDEV_Y+MEAN_Y)/(num_filt_timesteps*num_filt_features*STDDEV_X_filtsize) 
                '''
                core_score = np.transpose(np.multiply(X_rev[subjectID,t-int(filters.shape[0]/2):t+int(filters.shape[0]/2),:],filters[:,:,0]))*wdense/STDDEV_X_filtsize
                Wf_Wd_meanx = -filters[:,:,0].transpose()*wdense*MEAN_X_filtsize/STDDEV_X_filtsize
                Bf_Wd = biases.flatten()*wdense
                offset_score = (Wf_Wd_meanx + Bf_Wd + bdense)/(num_filt_timesteps*num_filt_features) 
                #print(score.shape, core_score.shape, STDDEV_X.shape, MEAN_X.shape, biases.shape, wdense.shape, bdense.shape)
                #       (43, 24)       (43, 24)         (43,)           (43,)           (1,)        (1,)        (1,)
                #print( Wf_Wd_meanx.shape, Bf_Wd.shape, bdense, MEAN_X_filtsize.shape)
                # (1032,) (1032,) (1032,) 
                '''
                '''            
                core_score = np.transpose(np.multiply(X[subjectID,t-int(filters.shape[0]/2):t+int(filters.shape[0]/2),:],filters[:,:,0]))*wdense
                Wf_Wd_meanx = -filters[:,:,0].transpose()*wdense*MEAN_X_filtsize
                Bf_Wd_stdx = biases.flatten()*wdense*STDDEV_X_filtsize
                Bd_stdx = bdense*STDDEV_X_filtsize
                print(score.shape, core_score.shape, STDDEV_X.shape, MEAN_X.shape, biases.shape, wdense.shape, bdense.shape)
                #       (43, 24)       (43, 24)         (43,)           (43,)           (1,)        (1,)        (1,)
                print( Wf_Wd_meanx.shape, Bf_Wd_stdx.shape, Bd_stdx.shape, MEAN_X_filtsize.shape)
                # (1032,) (1032,) (1032,) 
                offset_score = (Wf_Wd_meanx+Bf_Wd_stdx+Bd_stdx+MEAN_X_filtsize)/(num_filt_timesteps*num_filt_features)             
                #(biases.flatten()*STDDEV_X_filtsize*wdense[0]+bdense[0]*STDDEV_X_filtsize+MEAN_X_filtsize)/(num_filt_timesteps*num_filt_features)
                '''       
                score = core_score + offset_score                 
                #print(Y[subjectID,t], Y_rev[subjectID,t], np.sum(score), np.sum(core_score), np.sum(offset_score) ) 
                score2 = score    
            else:
                # Original score
                pass
                '''                
                print("using original score calculation")
                score=np.transpose(np.multiply(testtX[subjectID,t-int(filters.shape[0]/2):t+int(filters.shape[0]/2),:],filters[:,:,0]))*wdense + (biases.flatten()*wdense+bdense)/(filters.shape[0]*filters.shape[1])
                beta,meanC=[0,0]
                stdC,gamma=[1,1]
                #MEAN_X_filtsize=np.zeros_like(MEAN_X_filtsize)
                #STDDEV_X_filtsize=np.ones_like(STDDEV_X_filtsize)
                STDDEV_Y=1
                MEAN_Y=0
                divideby1 = STDDEV_X_filtsize*stdC
                divideby2 = num_filt_timesteps*num_filt_features
                weight_filter = filters[:,:,0]
                curr_X = testtX_orig[subjectID,t-int(filters.shape[0]/2):t+int(filters.shape[0]/2),:]

                core_score = np.transpose(  np.multiply(curr_X, weight_filter )   )
                core_score = core_score*gamma*wdense*STDDEV_Y/(STDDEV_X_filtsize*stdC)
                offset_score = -1 * np.transpose(  np.multiply( MEAN_X_filtsize.transpose(), weight_filter )   ) 
                offset_score = offset_score*gamma*wdense*STDDEV_Y/(STDDEV_X_filtsize*stdC)
                offset_score = offset_score + biases.flatten()*gamma*wdense*STDDEV_Y/(stdC)
                offset_score = offset_score - meanC*gamma*wdense*STDDEV_Y/(stdC)
                offset_score = offset_score + beta*wdense*STDDEV_Y
                offset_score = offset_score + bdense*STDDEV_Y
                offset_score = offset_score + MEAN_Y
                offset_score = offset_score/divideby2
                score2 = core_score+offset_score
                '''
            # Modified Score because X was scaled
            # np.transpose(np.multiply(X[subjectID,t-int(filters.shape[0]/2):t+int(filters.shape[0]/2),:],filters[:,:,0])).shape,(biases.flatten()*wdense+bdense).shape)
            #print((biases.flatten()*STDDEV_X*wdense+bdense*MEAN_X+MEAN_X).shape)
            plotmean=1
            if not plotmean:            
                # print(score.shape)
                hmap=sns.heatmap(score[:,:],cmap="PiYG", yticklabels=columnNames[:], #list(np.arange(score.shape[0])),
                xticklabels=4,
                linewidths=0.1, linecolor='gray', ax=ax1,
                center=0.00,
                 cbar=False)
                bottom, top = hmap.get_ylim()
                hmap.set_ylim(bottom + 0.5, top - 0.5)
            else:
                score=np.vstack((score,np.expand_dims(score.mean(axis=0),0)))
                score=np.hstack((score,np.expand_dims(score.mean(axis=1),1)))
                score[-1,-1]=0
                hmap=sns.heatmap(score[:,:],cmap="PiYG", yticklabels=columnNames[:]+['Mean W_t'], #list(np.arange(score.shape[0])),
                xticklabels=4,
                linewidths=0.1, linecolor='gray', ax=ax1,
                center=0.00,
                 cbar=False)
                '''
                locs, labels = plt.xticks()#ax1.get_xticklabels()
                labels[-1] = 'Mean W_f'
                print(locs,labels)
                hmap.set_xticklabels(labels=labels)
                '''
            bottom, top = hmap.get_ylim()
            hmap.set_ylim(bottom + 0.5, top - 0.5)
            #print(np.arange(t-filters.shape[0],t+filters.shape[0]).shape)
            #print(Y[subjectID,t-filters.shape[0]:t+filters.shape[0],0].shape)
            time_range=np.arange(t-neighbourhood_by_2+shiftinsamples,t+neighbourhood_by_2+shiftinsamples+1).flatten()

            sns.lineplot(x=time_range,  y=Y_rev[subjectID,time_range].flatten(),ax=ax2, label='Gold Standard')
            sns.lineplot(x=time_range, y=pY_rev[subjectID,time_range].flatten(),ax=ax2, label='Prediction')
            #sns.lineplot(x=time_range, y=pY[subjectID,time_range].flatten(),ax=ax2, label='Prediction_model')
            if ax2.get_ylim()[1]-ax2.get_ylim()[0]<0.5:
                midy=(ax2.get_ylim()[1]+ax2.get_ylim()[0])/2.0
                ax2.set_ylim((midy-0.25,midy+0.25))
            rect = patches.Rectangle((t+shiftinsamples-filters.shape[0]/2,ax2.get_ylim()[0]),filters.shape[0],ax2.get_ylim()[1]-ax2.get_ylim()[0],linewidth=1,edgecolor='k',facecolor='none')
            # sns.scatterplot(t,pY[subjectID,t],alpha=0.3, color="darkred", ax=ax2)
            if not plotmean:
                sns.scatterplot([t+shiftinsamples-1],[np.sum(score[:,:])], color="black", ax=ax2)#, scatter_kws={"s": 100})
                sns.scatterplot([t+shiftinsamples-1],[np.sum(score2[:,:])], color="red", ax=ax2)# scatter_kws={"s": 50})
            else:
                #sns.scatterplot([t+shiftinsamples-1],[np.sum(score[:-2,:-2])], color="black", ax=ax2)#, scatter_kws={"s": 100})
                sns.scatterplot([t+shiftinsamples-1],[np.sum(score2[:,:])], color="red", ax=ax2)# scatter_kws={"s": 50})
            #sns.scatterplot([t+shiftinsamples,t+shiftinsamples],[np.sum(score[:,:]), np.sum(score2[:,:])], 
            #                    palette=["black", "yellow"], size=[2,1], ax=ax2, legend=False)
            ax2.add_patch(rect)
            plt.tight_layout()
            plt.pause(0.001)  
            #plt.show()
          except Exception as E:
            print(E)
            pass
        else:
            while pause:
                fig1.canvas.get_tk_widget().update()
        fig1.canvas.get_tk_widget().update()



# normalize filter values to 0-1 so we can visualize them
# f_min, f_max = filters.min(), filters.max()
# filters = (filters - f_min) / (f_max - f_min)
ntsteps,nfeats,nfilters=filters.shape
num_rows = int(np.floor(np.sqrt(nfeats)))
num_cols = int(np.ceil(float(nfeats)/num_rows)) #int(math.ceil(tot_plots / float(num_cols)))
print(num_rows,num_cols)
# columnNames=[   'confidence_std', 'AU01_std','AU02_std','AU04_std',
#                 'AU05_std','AU06_std', 'AU07_std', 'AU09_std',
#                 'AU10_std','AU12_std', 'AU14_std', 'AU15_std',
#                 'AU17_std','AU20_std', 'AU23_std', 'AU25_std',
#                 'AU26_std','AU45_std', 'Turns'  ]

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(8, 10))
for idx, curFeat in enumerate(columnNames):
    axes.flatten()[idx].plot(filters[:,idx,0],label=columnNames[idx])
    axes.flatten()[idx].legend()

plt.show()

            # hmap=ax1.imshow(score,cmap="PiYG")
            # sns.lineplot(x=np.arange(t,t+filters.shape[0]), y=trainY[0][5,t:t+filters.shape[0],0],ax=ax1)
            # ax2.plot(np.arange(t-filters.shape[0]/2,t+filters.shape[0]+filters.shape[0]/2), Y[0][subjectID,t-filters.shape[0]/2:t+filters.shape[0]+filters.shape[0]/2,0],'k')
            # ax2.plot(np.arange(t-filters.shape[0]/2,t+filters.shape[0]+filters.shape[0]/2), pY[subjectID,t-filters.shape[0]/2:t+filters.shape[0]+filters.shape[0]/2],'b')
            # ax3.set_ylim([-0.5,0.8])
            # ax2.set_ylim([-0.5,0.8])

            # ax3.scatter(t+filters.shape[0]/2,np.sum(score))
            # cbar=plt.colorbar(hmap)
            # Note that using time.sleep does *not* work here!
            # except:
            # dgsd
            # pass
            # # fig1.canvas.draw()
            # # plt.cla()
            # # hmap.axes.clear()
            # ax3.clear()
            # # ax1.legend.remove()
            # ax1.clear()
            # cbar.remove()
            # plt.clf()

            # fig1.clear()
            # sns.reset_orig()
# fig = plt.figure()
# ax = plt.axes(xlim=(0,filters.shape[0]), ylim=(0,filters.shape[1]))
#
# def animate(i):
#     score=np.transpose(np.multiply(testCX[34,t:t+filters.shape[0],:],filters[:,:,0]))*wdense + (biases.flatten()*wdense+bdense)/(filters.shape[0]*filters.shape[1])
#     line.set_data(x, y)
#     return line,
# fig, ax = plt.subplots()

# def animate_heat_map():
#     fig = plt.figure()
#
#     def init():
#         plt.clf()
#         # ax = sns.heatmap(data, vmin=0, vmax=1)
#
#     def animate(t):
#         score=np.transpose(np.multiply(testCX[34,t:t+filters.shape[0],:],filters[:,:,0]))*wdense + (biases.flatten()*wdense+bdense)/(filters.shape[0]*filters.shape[1])
#         plt.clf()
#         ax = sns.heatmap(score)
#
#     anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1000)
#
#     plt.show()

# plt.figure()

# print('Shifting labels to the front for ' + str(shift_sec) + ' seconds.')

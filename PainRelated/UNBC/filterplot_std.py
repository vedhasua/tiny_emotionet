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

def shift_annotations_to_front(labels, shift=0):
    labels = np.concatenate((labels[:,shift:,:], np.zeros((labels.shape[0],shift,labels.shape[2]))), axis=1)
    return labels

def shift_annotations_to_back(labels, shift=0):
    labels = np.concatenate((np.zeros((labels.shape[0],shift,labels.shape[2])), labels[:,:labels.shape[1]-shift,:]), axis=1)
    return labels

origf,stdf,meanf=1,0,0
trystdf=0 and origf
columnNames=[   #'time',
                'AU00',
                'AU04', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 
                'AU15', 
                'AU20', 'AU25', 'AU26', 
                'AU27',
                'AU43', 
                'AU50',             
]

shiftinsamples = 0
modelFile      = '/home/vedhas/workspace/EmoPain/UNbC/s42b128e4000d0.0.h5'  #s42b128e4000d0.0.h5'#s42b128e2000d0.3.h5

model = load_model(modelFile, custom_objects={'ccc_loss_2': ccc_loss_2})
print([layer.name for layer in model.layers])
filters, biases = model.layers[1].get_weights()
wdense, bdense = model.layers[2].get_weights()
wdense = wdense.flatten()
bdense = bdense.flatten()
print(filters.shape)
print(filters, biases)
print(wdense, bdense)
'''
fig=plt.figure(figsize=(8.5,7))
sns.heatmap(np.transpose(filters[:,:,0]),cmap="PiYG",yticklabels=columnNames,xticklabels=10,center=0.000)
plt.tight_layout()
plt.show()
'''
###########
# NEW PLOT
SimpleOne=np.ones((1,filters.shape[0],filters.shape[1]))
score=np.transpose(np.multiply(SimpleOne,filters[:,:,0]))*wdense + (biases.flatten()*wdense+bdense)/(filters.shape[0]*filters.shape[1])
score=score[:,:,0]
fig=plt.figure(figsize=(8.5,7))
g = sns.heatmap(score,cmap="PiYG",yticklabels=columnNames,xticklabels=10,center=0.000,linewidths=0.5,linecolor='gray')
g.set_yticklabels(g.get_yticklabels(), rotation = 0)
bottom, top = g.get_ylim()
g.set_ylim(bottom + 0.5, top - 0.5)
plt.tight_layout()
plt.show()
########
# DATA IMPORT
max_seq_len  = 683
num_features = 14

facs_dir = '/home/vedhas/workspace/EmoPain/UNbC/Sequence_Labels/FACS/'
pspi_dir = '/home/vedhas/workspace/EmoPain/UNbC/Sequence_Labels/PSPI/'

facs_list = [ os.path.basename(cur_path) for cur_path in sorted(glob.glob(facs_dir+'*.csv')) ]
pspi_list = [ os.path.basename(cur_path) for cur_path in sorted(glob.glob(pspi_dir+'*.csv')) ]

feature_matrix  = np.zeros((200, max_seq_len, num_features ))
X,      Y       = [ np.zeros((200, max_seq_len, num_features )), np.zeros((200, max_seq_len, 1)) ]
# trainX, trainY  = [ np.zeros((200, max_seq_len, num_features )), np.zeros((200, max_seq_len, 1)) ]
# testX,  testY   = [ np.zeros((200, max_seq_len, num_features )), np.zeros((200, max_seq_len, 1)) ]

total_vector_count=[]
total_zero_count=[]
orig_Y=[]
for cur_seq_id, cur_seq_name in enumerate(facs_list):
    cur_facs = np.loadtxt(facs_dir + cur_seq_name, delimiter=',')
    cur_zero_count = cur_facs.shape[0]-np.count_nonzero(cur_facs[:, 1:].sum(axis=1))
    cur_pspi = np.loadtxt(pspi_dir + cur_seq_name)
    cur_pspi = np.expand_dims(cur_pspi,1)
    orig_Y.append(cur_pspi)
    total_vector_count.append(cur_facs.shape[0])
    total_zero_count.append(cur_zero_count)
    X[cur_seq_id, :cur_facs.shape[0], :] = cur_facs[:, 1:]
    Y[cur_seq_id, :cur_pspi.shape[0], :] = cur_pspi

# DATA IMPORT DONE
#####

from matplotlib import animation

if trystdf:
    filters=filters[:,list(np.arange(1,trainX.shape[2],2))+[-1],:]
    cgf
# trainY[0] = shift_annotations_to_back(trainY[0], shiftinsamples)
# develY[0] = shift_annotations_to_back(develY[0], shiftinsamples)
# testY[0]  = shift_annotations_to_back(testY[0],  shiftinsamples)

testX=X
testY=Y
if not trystdf:
    pY = model.predict(X)
else:
    model2=convert_to_stdf(model)
    predY=model2.predict(trainX)

pY  = shift_annotations_to_back(pY,  shiftinsamples)
num_samples_down = 1768

CCC_orig     = np.zeros(pY.shape[2])
CCC_orig_seq = np.zeros(pY.shape[2])
pred_list    = []  # required by label postprocessing script
orig_list    = []  # required by label postprocessing script - only one target

CCC_seq = []
predAll = np.array([])
origAll = np.array([])

for subjectID in range(X.shape[0]):
    #print(compute_ccc(Y[subjectID,:,:].flatten(), pY[subjectID,:,:].flatten()))
    predDown=pY[subjectID,:]#scipy.signal.resample(pY[subjectID,:],num_samples_down)
    lenOrig = len(orig_Y[subjectID])
    if len(predDown)>lenOrig:
        predDown = predDown[:lenOrig]
    elif len(predDown)<lenOrig:
        predDown = np.concatenate((predDown,np.zeros(lenOrig-len(predDown))))
    # segment avg eval
    CCC      = compute_ccc(predDown.flatten(),orig_Y[subjectID].flatten())
    CCC_seq.append(CCC)
    # global eval
    predAll  = np.append(predAll,predDown)
    origAll  = np.append(origAll,orig_Y[subjectID])
    # append to lists
    pred_list.append(predDown.flatten())
    orig_list.append(orig_Y[subjectID].flatten())
    #print(CCC_seq)
    # plt.plot(pY[subjectID,:].flatten(),'r')
    # plt.plot(predDown.flatten(),'g')
    # plt.plot(orig_Y[subjectID].flatten(),'b')
    # plt.plot(Y[0][subjectID],'k')
    # plt.show()

CCC_orig     = compute_ccc(predAll,origAll)  # global
print(CCC_seq)
print(CCC_orig)

#     # if True:
#     # subjectID=0
#     plt.figure()
#     plt.plot(Y[0][subjectID,:,:])
#     plt.plot(pY[subjectID,:])
#     plt.show()
# sda
# fig2,ax2=plt.subplots(1,1)
# sns.heatmap(np.zeros((filters.shape[0],filters.shape[1])),cmap="PiYG",ax=ax1)
# sns.lineplot(np.arange(0,0+filters.shape[0]), np.zeros(0+filters.shape[0]),ax=ax2)
pause = False
def press(event):
    global pause
    pause = not pause

import matplotlib.patches as patches


fig1,(ax1,ax2)=plt.subplots(2,1,figsize=(8,7),gridspec_kw={'height_ratios': [4, 1]})
fig1.canvas.mpl_connect('key_press_event', press)

plt.tight_layout()
neighbourhood_by_2=70#int(1.5*filters.shape[0])
print(filters.shape)
print(biases.shape)
print(wdense.shape)
print(bdense.shape)
print("-------------")
for subjectID in np.array([1, 5, 7, 8, 9, 11, 14, 16, 17, 22, 24, 26, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 41])[14:15]: #range(X.shape[0]):
    for t in range(105,130,1):
        if not pause and t-int(filters.shape[0]/2)>0 and t+int(filters.shape[0]/2)<total_vector_count[subjectID]:
          try:
            #print(np.multiply(X[subjectID,t-int(filters.shape[0]/2):t+int(filters.shape[0]/2),:],filters[:,:,0]))
            ax1.clear()
            ax2.clear()
            score=np.transpose(np.multiply(X[subjectID,t-int(filters.shape[0]/2):t+int(filters.shape[0]/2),:],filters[:,:,0]))*wdense + (biases.flatten()*wdense+bdense)/(filters.shape[0]*filters.shape[1])
            #print(score.shape)
            hmap=sns.heatmap(score,cmap="PiYG", yticklabels=columnNames, #list(np.arange(score.shape[0])),
            xticklabels=5,
            linewidths=0.1, linecolor='white', ax=ax1,
            center=0.00,
             cbar=False)
            #print(np.arange(t-filters.shape[0],t+filters.shape[0]).shape)
            #print(Y[subjectID,t-filters.shape[0]:t+filters.shape[0],0].shape)
            sns.lineplot(x=np.arange(t-neighbourhood_by_2+shiftinsamples,t+neighbourhood_by_2+shiftinsamples+1).flatten(),
             y=Y[subjectID,t-neighbourhood_by_2+shiftinsamples:t+neighbourhood_by_2+shiftinsamples+1,0].flatten(),ax=ax2, label='Gold Standard')
            sns.lineplot(x=np.arange(t-neighbourhood_by_2+shiftinsamples,t+neighbourhood_by_2+shiftinsamples+1).flatten(),
             y=  pY[subjectID,t-neighbourhood_by_2+shiftinsamples:t+neighbourhood_by_2+shiftinsamples+1].flatten(),ax=ax2, label='Prediction')
            if ax2.get_ylim()[1]-ax2.get_ylim()[0]<0.5:
                midy=(ax2.get_ylim()[1]+ax2.get_ylim()[0])/2.0
                ax2.set_ylim((midy-0.25,midy+0.25))
            rect = patches.Rectangle((t+shiftinsamples-filters.shape[0]/2,ax2.get_ylim()[0]),filters.shape[0],ax2.get_ylim()[1]-ax2.get_ylim()[0],linewidth=1,edgecolor='k',facecolor='none')
            # sns.scatterplot(t,pY[subjectID,t],alpha=0.3, color="darkred", ax=ax2)
            sns.scatterplot([t+shiftinsamples-1],[np.sum(score)], color="black", ax=ax2)
            ax2.add_patch(rect)
            plt.tight_layout()
            #plt.pause(1)  
            plt.show()
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

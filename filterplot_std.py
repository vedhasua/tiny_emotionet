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
from modelconversion import convert_to_stdf
from ccc import compute_ccc

# def shift_annotations_to_front(labels, shift=0):
#     labels = np.concatenate((labels[:,shift:,:], np.zeros((labels.shape[0],shift,labels.shape[2]))), axis=1)
#     return labels
#
# def shift_annotations_to_back(labels, shift=0):
#     labels = np.concatenate((np.zeros((labels.shape[0],shift,labels.shape[2])), labels[:,:labels.shape[1]-shift,:]), axis=1)
    # return labels

origf,stdf,meanf=1,0,0
trystdf=0 and origf
columnNames=[   'confidence_mean','confidence_std', 'AU01_mean','AU01_std', 'AU02_mean','AU02_std',
                'AU04_mean',        'AU04_std',     'AU05_mean','AU05_std', 'AU06_mean','AU06_std',
                'AU07_mean',        'AU07_std',     'AU09_mean','AU09_std', 'AU10_mean','AU10_std',
                'AU12_mean',        'AU12_std',     'AU14_mean','AU14_std', 'AU15_mean','AU15_std',
                'AU17_mean',        'AU17_std',     'AU20_mean','AU20_std', 'AU23_mean','AU23_std',
                'AU25_mean',        'AU25_std',     'AU26_mean','AU26_std', 'AU45_mean','AU45_std',
                'Turns'    ,]


modelFile          = '/home/vedhas/workspace/is2019_recurrence/DHCresults/8/m531_tr0.646_dv0.600_ts0.602_tc0.413.h5'
# '/home/vedhas/workspace/is2019_recurrence/DHCresults/6/m619_tr0.657_dv0.595_ts0.603_tc0.423.h5'
# '/home/vedhas/workspace/is2019_recurrence/DHCresults/8/m531_tr0.646_dv0.600_ts0.602_tc0.413.h5'
# '/home/vedhas/workspace/is2019_recurrence/DHCresults/5/m1191_tr0.649_dv0.602_ts0.602_tc0.415.h5'
# '/home/vedhas/workspace/is2019_recurrence/DHCresults/7/m457_tr0.595_dv0.599_ts0.606_tc0.525.h5'
# '/home/vedhas/workspace/is2019_recurrence/DHCresults/6/m619_tr0.657_dv0.595_ts0.603_tc0.423.h5'
#'/home/vedhas/workspace/is2019_recurrence/DHCresults/2/m1000_tr0.687_dv0.608_ts0.625_tc0.420.h5'
# '/home/vedhas/workspace/is2019_recurrence/DHCresults/2/m1000_tr0.687_dv0.608_ts0.625_tc0.420.h5'
#'/home/vedhas/workspace/is2019_recurrence/DHCresults/stdonly/1/m947_tr0.659_dv0.588_ts0.611_tc0.422.h5' #'/home/vedhas/workspace/is2019_recurrence/DHCresults/2/m1000_tr0.687_dv0.608_ts0.625_tc0.420.h5'
shiftinsamples = 250
model = load_model(modelFile, custom_objects={'ccc_loss_2': ccc_loss_2})
print([layer.name for layer in model.layers])
filters, biases = model.layers[1].get_weights()
wdense, bdense = model.layers[2].get_weights()
wdense = wdense.flatten()
bdense = bdense.flatten()
print(filters.shape)
fig=plt.figure(figsize=(8.5,7))
sns.heatmap(np.transpose(filters[:,:,0]),cmap="PiYG",yticklabels=columnNames,xticklabels=10,center=0.000)
plt.tight_layout()
plt.show()

feature_type_v='faus'
targets_avl='V'
fsuffix="_v_{}".format(feature_type_v)

# Targets
targets = []
if targets_avl.find('A')>=0: targets.append(0)
if targets_avl.find('V')>=0: targets.append(1)
if targets_avl.find('L')>=0: targets.append(2)

DeNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/DeAVL{}.npy'.format(fsuffix)
HuNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/HuAVL{}.npy'.format(fsuffix)
CnNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/CnAVL{}.npy'.format(fsuffix)

if os.path.isfile(DeNPY):
    trainX, trainY, develX, develY, testX, testY, origTrain, origDevel, origTest = np.load(DeNPY,allow_pickle=True, encoding = 'bytes')
    trainY, develY, testY, origTrain, origDevel, origTest = [[curList[targets[0]]] for curList in [trainY, develY, testY, origTrain, origDevel, origTest] ]
if os.path.isfile(HuNPY):
    trainHX, trainHY, develHX, develHY, testHX, testHY, origHTrain, origHDevel, origHTest, origCross, crossX, crossY = np.load(HuNPY,allow_pickle=True, encoding = 'bytes')
    trainHY, develHY, testHY, origHTrain, origHDevel, origHTest, origCross, crossY = [[curList[targets[0]]] for curList in [trainHY, develHY, testHY, origHTrain, origHDevel, origHTest, origCross, crossY] ]
if os.path.isfile(CnNPY):
    trainCX, trainCY, develCX, develCY, testCX, testCY, origCTrain, origCDevel, origCTest, origCrossC, crossXC, crossYC = np.load(CnNPY,allow_pickle=True, encoding = 'bytes')
    # trainCY, develCY, testCY, origCTrain, origCDevel, origCTest, origCrossC, crossYC = [[curList[targets[0]]] for curList in [trainCY, develCY, testCY, origCTrain, origCDevel, origCTest, origCrossC, crossYC] ]

if stdf or trystdf:
    trainX = trainX[:,:,list(np.arange(1,trainX.shape[2],2))+[-1]]
    develX = develX[:,:,list(np.arange(1,develX.shape[2],2))+[-1]]
    testX  = testX[:,:,list(np.arange(1,testX.shape[2],2))+[-1]]
    trainHX = trainHX[:,:,list(np.arange(1,trainHX.shape[2],2))+[-1]]
    develHX = develHX[:,:,list(np.arange(1,develHX.shape[2],2))+[-1]]
    testHX  = testHX[:,:,list(np.arange(1,testHX.shape[2],2))+[-1]]
    crossX = np.concatenate((trainHX, develHX, testHX), axis=0)
    trainCX = trainCX[:,:,list(np.arange(1,trainCX.shape[2],2))+[-1]]
    develCX = develCX[:,:,list(np.arange(1,develCX.shape[2],2))+[-1]]
    testCX  = testCX[:,:,list(np.arange(1,testCX.shape[2],2))+[-1]]
    crossXC  = crossXC[:,:,list(np.arange(1,crossXC.shape[2],2))+[-1]]
if meanf:
    trainX = trainX[:,:,list(np.arange(0,trainX.shape[2],2))]
    develX = develX[:,:,list(np.arange(0,develX.shape[2],2))]
    testX  = testX[:,:,list(np.arange(0,testX.shape[2],2))]
    trainHX = trainHX[:,:,list(np.arange(0,trainHX.shape[2],2))]
    develHX = develHX[:,:,list(np.arange(0,develHX.shape[2],2))]
    testHX  = testHX[:,:,list(np.arange(0,testHX.shape[2],2))]
    crossX = np.concatenate((trainHX, develHX, testHX), axis=0)
    trainCX = trainCX[:,:,list(np.arange(0,trainCX.shape[2],2))]
    develCX = develCX[:,:,list(np.arange(0,develCX.shape[2],2))]
    testCX  = testCX[:,:,list(np.arange(0,testCX.shape[2],2))]
    crossXC  = crossXC[:,:,list(np.arange(0,crossXC.shape[2],2))]

from matplotlib import animation

if trystdf:
    filters=filters[:,list(np.arange(1,trainX.shape[2],2))+[-1],:]
    cgf
# trainY[0] = shift_annotations_to_front(trainY[0], shiftinsamples)
# develY[0] = shift_annotations_to_front(develY[0], shiftinsamples)
# testY[0]  = shift_annotations_to_front(testY[0],  shiftinsamples)

X=testX
Y=testY
if not trystdf:
    pY = model.predict(X)
else:
    model2=convert_to_stdf(model)
    predY=model2.predict(trainX)

# for subjectID in range(0,X.shape[0]):
#     print(compute_ccc(Y[0][subjectID,:,:].flatten(), pY[subjectID,:].flatten()))
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


fig1,(ax1,ax2)=plt.subplots(2,1,figsize=(8,7),gridspec_kw={'height_ratios': [7, 1]})
fig1.canvas.mpl_connect('key_press_event', press)

plt.tight_layout()
neighbourhood_by_2=int(1.5*filters.shape[0])

for subjectID in range(0,X.shape[0]):
    for t in range(neighbourhood_by_2,1700,10):
        if not pause:
          try:
            ax1.clear()
            ax2.clear()
            score=np.transpose(np.multiply(X[subjectID,t-filters.shape[0]/2:t+filters.shape[0]/2,:],filters[:,:,0]))*wdense + (biases.flatten()*wdense+bdense)/(filters.shape[0]*filters.shape[1])
            # print(score.shape)
            hmap=sns.heatmap(score,cmap="PiYG", yticklabels=columnNames, #list(np.arange(score.shape[0])),
            xticklabels=5,
            linewidths=0.1, linecolor='white', ax=ax1,
            center=0.00,
             cbar=False)
            # print(np.arange(t-filters.shape[0],t+filters.shape[0]).shape)
            # print(Y[0][subjectID,t-filters.shape[0]:t+filters.shape[0],0].shape)

            sns.lineplot(x=np.arange(t-neighbourhood_by_2,t+neighbourhood_by_2).flatten(),
             y=Y[0][subjectID,t-neighbourhood_by_2:t+neighbourhood_by_2,0].flatten(),ax=ax2, label='Gold Standard')
            sns.lineplot(x=np.arange(t-neighbourhood_by_2,t+neighbourhood_by_2).flatten(),
            y=  pY[subjectID,t-neighbourhood_by_2:t+neighbourhood_by_2].flatten(),ax=ax2, label='Prediction')
            if ax2.get_ylim()[1]-ax2.get_ylim()[0]<0.5:
                midy=(ax2.get_ylim()[1]+ax2.get_ylim()[0])/2.0
                ax2.set_ylim((midy-0.25,midy+0.25))
            rect = patches.Rectangle((t-filters.shape[0]/2,ax2.get_ylim()[0]),filters.shape[0],ax2.get_ylim()[1]-ax2.get_ylim()[0],linewidth=1,edgecolor='k',facecolor='none')
            # sns.scatterplot(t,pY[subjectID,t],alpha=0.3, color="darkred", ax=ax2)
            sns.scatterplot([t],[np.sum(score)],alpha=0.3, color="darkred", ax=ax2)
            ax2.add_patch(rect)
            plt.tight_layout()
            plt.pause(0.0001)
            ffa
          except:
            pass
        else:
            while pause:
                fig1.canvas.get_tk_widget().update()
        fig1.canvas.get_tk_widget().update()
sds


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

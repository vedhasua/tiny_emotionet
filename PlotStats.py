import os
import pandas as pd
import glob
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc("lines", markeredgewidth=0.5)
matplotlib.rc('font', weight='bold')
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

from sewa_data_continued import load_labels
sns.set_style("dark")
sns.set_context("notebook",rc={"lines.linewidth": 1.2})
# sns.despine()

Cultures=['German','Hungarian','Chinese']
fsuffix='_v_faus'
DeNPY   = '/home/vedhas/workspace/is2019_recurrence/NPY/De{}.npy'.format(fsuffix)
HuNPY   = '/home/vedhas/workspace/is2019_recurrence/NPY/Hu{}.npy'.format(fsuffix)
CnNPY   = '/home/vedhas/workspace/is2019_recurrence/NPY/Cn{}.npy'.format(fsuffix)
NpyPath={Cultures[0]:DeNPY, Cultures[1]:HuNPY, Cultures[2]:CnNPY, }
num_train={Cultures[0]:34, Cultures[1]:34, Cultures[2]:0, }
num_devel={Cultures[0]:14, Cultures[1]:14, Cultures[2]:0, }
num_test={Cultures[0]:16, Cultures[1]:18, Cultures[2]:70, }

overallN = np.zeros((0,40),dtype='float')
for cid, culture in enumerate(Cultures):
    if os.path.isfile(NpyPath[culture]) and cid==0:
        trainX, trainY, develX, develY, testX, testY, origTrain, origDevel, origTest = np.load(NpyPath[culture],allow_pickle=True)
    else:
        trainX, trainY, develX, develY, testX, testY, origTrain, origDevel, origTest, origCross, crossX, crossY =  np.load(NpyPath[culture],allow_pickle=True)

    # print(trainX.shape,develX.shape, testX.shape)
    crossN = np.concatenate((trainX, develX, testX), axis=0)
    crossN = crossN.reshape(-1,crossN.shape[-1])
    crossN = np.concatenate((crossN, cid*np.ones((crossN.shape[0],1)) ), axis=1)

    labelsTrain, trainY = load_labels(culture, 'Train', num_train[culture], 1768, 1, targets=[0,1])
    labelsDevel, develY = load_labels(culture, 'Devel', num_devel[culture], 1768, 1, targets=[0,1])
    labelsTest,  testY  = load_labels(culture, 'Test',  num_test[culture],  1768, 1, targets=[0,1])


    # trainY, develY, testY = np.array(trainY).reshape(-1,1), np.array(develY).reshape(-1,1), np.array(testY).reshape(-1,1)
    trainY, develY, testY = np.array(trainY).squeeze(), np.array(develY).squeeze(), np.array(testY).squeeze()
    trainY, develY, testY = np.moveaxis(trainY,0,-1), np.moveaxis(develY,0,-1), np.moveaxis(testY,0,-1)
    trainY, develY, testY = trainY.reshape(-1,trainY.shape[-1]), develY.reshape(-1,develY.shape[-1]), testY.reshape(-1,testY.shape[-1])
    # print(trainX.shape, trainY.shape)
    # print(trainY.shape,develY.shape, testY.shape)
    labelN = np.concatenate((trainY, develY, testY), axis=0)
    labelN = labelN.reshape(-1,labelN.shape[-1])
    # print(crossN.shape,labelN.shape)

    crossN = np.concatenate((crossN, labelN), axis=1)
    overallN = np.concatenate((overallN,crossN), axis=0)

columnNames=["feat{}".format(i) for i in range(38)]
columnNames=[   'confidence_mean','confidence_std', 'AU01_mean','AU01_std', 'AU02_mean','AU02_std', 'AU04_mean','AU04_std',
                'AU05_mean','AU05_std', 'AU06_mean','AU06_std', 'AU07_mean','AU07_std', 'AU09_mean','AU09_std',
                'AU10_mean','AU10_std', 'AU12_mean','AU12_std', 'AU14_mean','AU14_std', 'AU15_mean','AU15_std',
                'AU17_mean','AU17_std', 'AU20_mean','AU20_std', 'AU23_mean','AU23_std', 'AU25_mean','AU25_std',
                'AU26_mean','AU26_std', 'AU45_mean','AU45_std', 'Turns'    ,'Culture',  'Arousal',  'Valence']

df = pd.DataFrame(overallN, columns=columnNames[:40])
df['Culture'] = df['Culture'].map({0.00: 'German', 1.00: 'Hungarian', 2.00:'Chinese'})
medvals=df.median(axis=0).to_numpy()
norm = plt.Normalize()
color = plt.cm.Blues(norm(medvals))
dd=pd.melt(df,id_vars=['Culture'], value_vars=columnNames[:37], var_name='features')
# sns.boxplot(y='value', x='features', data=dd, hue='Culture')
# plt.show()

num_cols = 4
num_rows = 5 #int(math.ceil(tot_plots / float(num_cols)))

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(16, 16))
flierprops = dict(marker='o', markersize=0.5)

for idx, ax in enumerate(axes.flatten()):
    if idx<18:
        value_vars=[columnNames[i] for i in [idx*2,idx*2+1]]
        # ax.set_facecolor(color[idx//2])
    elif idx==18:
        value_vars=[columnNames[i] for i in [38]]
        # ax.set_facecolor(color[-2])
    elif idx==19:
        value_vars=[columnNames[i] for i in [39]]
        # ax.set_facecolor(color[-1])

    dd=pd.melt(df[value_vars+['Culture']],id_vars=['Culture'], value_vars=value_vars, var_name='features')
    if idx<18:
        sns.boxplot(x='value', y='features', orient='h', data=dd, hue='Culture',ax=ax, notch=True, flierprops=flierprops,
        palette="hls") #Set2 is also somewhat okay, #sns.hls_palette(8, l=.3, s=.8))
    else:
        sns.violinplot(x='value', y='features', orient='h', data=dd, hue='Culture',ax=ax, notch=True, flierprops=flierprops,
        palette="hls") #Set2 is also somewhat okay, #sns.hls_palette(8, l=.3, s=.8))
    # g._legend.remove()
    ax.get_legend().remove()
    # ax.tick_params(direction="in")
    if True: #idx<18:
        # ax.tick_params(axis="x",direction="in", pad=-47.5)
        ax.tick_params(axis="x",direction="in")
    # locs, labels = plt.xticks()
    # labels = ax.get_xticks()
    # labels = labels.get_xticklabels() #[item.get_text() for item in ax.get_xticklabels()]
    # print(labels)
    # if float(labels[1])-float(labels[0])<0.5:
    #     ax.xaxis.set_ticks(np.arange(locs[0], locs[-1], 0.5))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_title(r'$\bf{}$'.format(value_vars[0].replace('_mean','')),position=(0.85,0.4), size=10)
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_ticklabels(['mean','std'])
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=-90)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(0.185, 1.0))
# fig.suptitle("Sampling BoxPlots", x=0.5, y=0.93, fontsize=14, fontweight="bold")
plt.tight_layout()
# fig.tight_layout(rect=[0.1,0.1,0.9, 0.95])
# plt.savefig("/home/vedhas/workspace/is2019_recurrence/LIT/images/feats.png", format="png")
# plt.subplots_adjust(top=0.8)
plt.show()








'''
df = pd.DataFrame(overallX, columns=columnNames[:37])

dd=pd.melt(df,id_vars=['feat37'], value_vars=columnNames[:37], var_name='features')
print(df)
print(dd)
sns.boxplot(y='value', x='features', data=dd, hue='feat37')
plt.show()
'''

#sns.heatmap(overallX[:,:-1])
#sns.boxplot(x=np.arange(0,overallX.shape[1]), y=overallX[:,6])#,
#            hue=overallX[:,-1], palette=["m", "g", 'b'])
# plt.show()

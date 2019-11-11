import os
import pandas as pd
import glob
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc("lines", markeredgewidth=0.5)
# matplotlib.rc('font', weight='bold')
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

from sewa_data_continued import load_labels
sns.set_style("dark")
sns.set_context("notebook",rc={"lines.linewidth": 1.2})
# sns.despine()

Cultures=['German','Hungarian','Chinese']
fsuffix='AVL_v_faus'
DeNPY   = '/home/vedhas/workspace/is2019_recurrence/NPY/De{}.npy'.format(fsuffix)
HuNPY   = '/home/vedhas/workspace/is2019_recurrence/NPY/Hu{}.npy'.format(fsuffix)
CnNPY   = '/home/vedhas/workspace/is2019_recurrence/NPY/Cn{}.npy'.format(fsuffix)
NpyPath={Cultures[0]:DeNPY, Cultures[1]:HuNPY, Cultures[2]:CnNPY, }
num_train={Cultures[0]:34, Cultures[1]:34, Cultures[2]:0, }
num_devel={Cultures[0]:14, Cultures[1]:14, Cultures[2]:0, }
num_test={Cultures[0]:16, Cultures[1]:18, Cultures[2]:70, }

overallN = np.zeros((0,41),dtype='float')
for cid, culture in enumerate(Cultures):
    if os.path.isfile(NpyPath[culture]) and cid==0:
        trainX, trainY, develX, develY, testX, testY, origTrain, origDevel, origTest = np.load(NpyPath[culture],allow_pickle=True)
    else:
        trainX, trainY, develX, develY, testX, testY, origTrain, origDevel, origTest, origCross, crossX, crossY =  np.load(NpyPath[culture],allow_pickle=True)

    print(len(testY),testY[0].shape)
    # print(trainX.shape,develX.shape, testX.shape)
    crossN = np.concatenate((trainX, develX, testX), axis=0)
    crossN = crossN.reshape(-1,crossN.shape[-1])
    crossN = np.concatenate((crossN, cid*np.ones((crossN.shape[0],1)) ), axis=1)

    labelsTrain, trainY = load_labels(culture, 'Train', num_train[culture], 1768, 1, targets=[0,1,2])
    labelsDevel, develY = load_labels(culture, 'Devel', num_devel[culture], 1768, 1, targets=[0,1,2])
    labelsTest,  testY  = load_labels(culture, 'Test',  num_test[culture],  1768, 1, targets=[0,1,2])


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
columnNames=[   'confidence_mean','confidence_std', 'AU01_mean','AU01_std', 'AU02_mean','AU02_std',
                'AU04_mean',        'AU04_std',     'AU05_mean','AU05_std', 'AU06_mean','AU06_std',
                'AU07_mean',        'AU07_std',     'AU09_mean','AU09_std', 'AU10_mean','AU10_std',
                'AU12_mean',        'AU12_std',     'AU14_mean','AU14_std', 'AU15_mean','AU15_std',
                'AU17_mean',        'AU17_std',     'AU20_mean','AU20_std', 'AU23_mean','AU23_std',
                'AU25_mean',        'AU25_std',     'AU26_mean','AU26_std', 'AU45_mean','AU45_std',
                'Turns'    ,        'Culture',  'Arousal',  'Valence', 'Liking']

heatxticklabels=['median','mean','std.dev.','min','max']
heatyticklabels=columnNames[0:36]+columnNames[-3:]
df = pd.DataFrame(overallN, columns=columnNames)
df['Culture'] = df['Culture'].map({0.00: 'German', 1.00: 'Hungarian', 2.00:'Chinese'})
df2=df[df['culture']==0.00]

print(len((df2['AU01_std'].to_numpy().nonzero()[0])))
print(len((df2['AU02_std'].to_numpy().nonzero()[0])))
print(len((df2['AU04_std'].to_numpy().nonzero()[0])))
print(len((df2['AU05_std'].to_numpy().nonzero()[0])))
print(len((df2['AU06_std'].to_numpy().nonzero()[0])))
print(len((df2['AU07_std'].to_numpy().nonzero()[0])))
print(len((df2['AU09_std'].to_numpy().nonzero()[0])))
print(len((df2['AU10_std'].to_numpy().nonzero()[0])))
print(df2.shape)
# 217462      1
# 242517      2
# 132296      4
# 255259      5
# 172624      6
# 139441      7
# 231592      9
# 171387      10
# (353600, 41)
fdd
medvals=df.median(axis=0,numeric_only=True).to_numpy()

print(medvals.shape)
print([
# '%.7f'%
i for i in medvals])
adfds
avgvals=df.mean(axis=0,numeric_only=True).to_numpy()
# print(avgvals.shape)
stdvals=df.std(axis=0).to_numpy()
# print(stdvals.shape)
minvals=df.min(axis=0,numeric_only=True).to_numpy()
# print(minvals.shape)
maxvals=df.max(axis=0,numeric_only=True).to_numpy()
# print(maxvals.shape)
heatmapstats=np.vstack((medvals,avgvals,stdvals,minvals,maxvals)).transpose()
print(heatmapstats.shape)
print(heatmapstats[36])
heatmapstats=np.delete(heatmapstats,36,0)
StatNames=["median","average","std","min","max"]
for curStatid, curStatvals in enumerate(heatmapstats.transpose()):
    print(StatNames[curStatid])
    print(['%.3f'%i for i in curStatvals])
print(heatmapstats.shape)
plt.figure(figsize=(16, 16))
ax=sns.heatmap(heatmapstats,linewidths=.5, xticklabels=heatxticklabels, yticklabels=heatyticklabels,square=True)
ax.xaxis.set_ticks_position('top')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
# plt.show()
# print(medvals)
# print(avgvals)
# print(stdvals)
# print(minvals)
# print(maxvals)
plt.tight_layout()
plt.subplots_adjust(top=0.905,bottom=0.005)
plt.show()
sf
fsdf

# Add a table at the bottom of the axes


# cell_text=medvals
# print(medvals)
# the_table = plt.table(cellText=cell_text,
#                       rowLabels=rows,
#                       rowColours=colors,
#                       colLabels=columns,
#                       loc='bottom')

# norm = plt.Normalize()
# color = plt.cm.Blues(norm(medvals))
# dd=pd.melt(df,id_vars=['Culture'], value_vars=columnNames[:37], var_name='features')
# sns.boxplot(y='value', x='features', data=dd, hue='Culture')
# plt.show()

num_cols = 3
num_rows = 7 #int(math.ceil(tot_plots / float(num_cols)))

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(8, 10))
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
    elif idx==20:
        value_vars=[columnNames[i] for i in [40]]

    dd=pd.melt(df[value_vars+['Culture']],id_vars=['Culture'], value_vars=value_vars, var_name='features')
    if idx<18:
        sns.boxplot(x='value', y='features', orient='h', data=dd, hue='Culture',ax=ax, notch=True, flierprops=flierprops,
        palette="hls") #Set2 is also somewhat okay, #sns.hls_palette(8, l=.3, s=.8))
    else:
        sns.violinplot(x='value', y='features', orient='h', data=dd, hue='Culture',ax=ax, notch=True, flierprops=flierprops,
        palette="hls") #Set2 is also somewhat okay, #sns.hls_palette(8, l=.3, s=.8))
    ax.get_legend().remove()
    ax.tick_params(axis="x",direction="in")
    ax.tick_params(axis="y",direction="in",pad=-2)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    if idx<18:
        ax.set_title(r'$\bf{}$'.format(value_vars[0].replace('_mean','')),loc='right', position=(1.0,0.7), size=10)
        ax.yaxis.set_ticklabels(['mean','std'])
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=30)
    else:
        ax.set_title(r'$\bf{}$'.format(value_vars[0].replace('_mean','')),loc='right', position=(1.0,0.7), size=10)
        ax.yaxis.set_ticklabels([''])
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=12)

handles, labels = ax.get_legend_handles_labels()
axes[3][2].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
                                # bbox_to_anchor=(0.185, 1.0))
# fig.suptitle("Sampling BoxPlots", x=0.5, y=0.93, fontsize=14, fontweight="bold")
plt.tight_layout()
plt.subplots_adjust(wspace=0.33, hspace=0.33)


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

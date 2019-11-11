import os
import pandas as pd
import glob
import numpy as np

FeatsDir='/home/vedhas/workspace/is2019_recurrence/Chinese/visual_features/'
Turn1Dir='/home/vedhas/workspace/is2019_recurrence/Chinese/turns/'
Turn2Dir='/home/vedhas/workspace/is2019_recurrence/Chinese/turn_features_chunk/'
LabelDir='/home/vedhas/workspace/is2019_recurrence/Chinese/labels/'

FeatsCsvList=sorted(glob.glob('{}/*.csv'.format(FeatsDir)))
Turn1CsvList=sorted(glob.glob('{}/*.csv'.format(Turn1Dir)))

for curcsv in Turn1CsvList:
    curname=os.path.basename(curcsv)
    curFeats=os.path.join(FeatsDir+curname)
    curTurn1=os.path.join(Turn1Dir+curname)
    curTurn2=os.path.join(Turn2Dir+curname)
    curLabel=os.path.join(LabelDir+curname)
    featsDta=pd.read_csv(curFeats,sep=';',header=0,usecols=[0,1]).to_numpy()
    LabelDta=pd.read_csv(curLabel,sep=';',header=0,usecols=[0,1]).to_numpy()
    #tmax=np.max(np.concatenate((LabelDta[:,1],featsDta[:,1])))
    tmax=176.71
    turn1Dta=pd.read_csv(curTurn1,sep=';',header=None).to_numpy()
    time2Dta=np.arange(0,tmax,0.1)
    turn2Dta=np.zeros_like(time2Dta)
    for tbeg, tend in turn1Dta:
        curmask=np.where((time2Dta>=tbeg) & (time2Dta<=tend))
        turn2Dta[curmask]=1.0
    time2Dta=np.vstack((time2Dta,turn2Dta)).transpose()
    np.savetxt(curTurn2,time2Dta,delimiter=';',fmt='%.2f',)

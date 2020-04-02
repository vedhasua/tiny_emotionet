import glob
import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


facs_dir = '/home/vedhas/workspace/EmoPain/UNbC/Sequence_Labels/FACS/'
pspi_dir = '/home/vedhas/workspace/EmoPain/UNbC/Sequence_Labels/PSPI/'

facs_list = [ os.path.basename(cur_path) for cur_path in sorted(glob.glob(facs_dir+'*.csv')) ]
pspi_list = [ os.path.basename(cur_path) for cur_path in sorted(glob.glob(pspi_dir+'*.csv')) ]

assert facs_list==pspi_list

timesteps=np.zeros((0,2))

for cur_seq_name in facs_list:
    cur_facs = np.loadtxt(facs_dir + cur_seq_name, delimiter=',')
    cur_pspi = np.loadtxt(pspi_dir + cur_seq_name)

    # Pain = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
    cur_comp_pspi = cur_facs[:,2] + np.max(cur_facs[:,3:5],1) + cur_facs[:,-2] + np.max(cur_facs[:,5:7],1)
    print("{}:\t{} timesteps\t{} Non-zero".format(cur_seq_name, cur_comp_pspi.shape[0], np.count_nonzero(cur_comp_pspi > 0)))
    timesteps=np.vstack((timesteps,[cur_comp_pspi.shape[0], np.count_nonzero(cur_comp_pspi > 0)]))

    if not (cur_pspi == cur_comp_pspi).all():
        print(cur_seq_name, cur_facs.shape, cur_pspi.shape, cur_comp_pspi.shape)
        print( np.where( (cur_pspi == cur_comp_pspi) == False ))

print(timesteps)
print(timesteps.sum(axis=0))
print(timesteps.max(axis=0))
print(timesteps.min(axis=0))
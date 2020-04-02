import glob
import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

dic={0:0,
     4:1,
     6:2,
     7:3,
     9:4,
     10:5,
     12:6,
     15:7,
     20:8,
     25:9,
     26:10,
     27:11,
     43:12,
     50:13}


def replace_with_dict(ar, dic):
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    # Drop the magic bomb with searchsorted to get the corresponding
    # places for a in keys (using sorter since a is not necessarily sorted).
    # Then trace it back to original order with indexing into sidx
    # Finally index into values for desired output.
    return v[sidx[np.searchsorted(k,ar,sorter=sidx)]]

out_root_dir = "/home/vedhas/workspace/EmoPain/UNbC/Sequence_Labels/"
out_facs_dir = out_root_dir + "FACS2/"
out_pspi_dir = out_root_dir + "PSPI/"

in_facs_dir = "/home/vedhas/workspace/EmoPain/UNbC/Frame_Labels/FACS/"
in_pspi_dir = "/home/vedhas/workspace/EmoPain/UNbC/Frame_Labels/PSPI/"

subject_list = [os.path.basename(cur_path) for cur_path in sorted(glob.glob(in_facs_dir + '*'))]

overall_np = np.zeros((0,5))
for cur_subject in subject_list:
    sequence_list = [os.path.basename(cur_path) for cur_path in sorted(glob.glob(in_facs_dir + cur_subject + '/*'))]
    for cur_seq in sequence_list:
        file_list = sorted(glob.glob(os.path.join(in_facs_dir , cur_subject, cur_seq)+'/*'))
        cur_val = []
        cur_row = []
        cur_col = []
        lst_rcv = np.zeros((0,5),dtype=int)
        cur_pspi_seq = np.zeros((len(file_list),1),dtype=int)
        for cur_file in file_list:
            if os.path.getsize(cur_file) > 0:
                cur_time = int(os.path.basename(cur_file).split('.')[0].split('_facs')[0][-3:])-1
                cur_df   = pd.read_csv(cur_file, sep='\s+', header=None, index_col=None,
                                       names = ['fau','intensity','onset','offset'],
                                       dtype={'fau':int,'intensity':int,'onset':int,'offset':int})
                cur_file_lbl = os.path.join(in_pspi_dir, cur_subject , cur_seq , os.path.basename(cur_file))
                cur_df_lbl = np.loadtxt(cur_file_lbl)
                cur_df.loc[cur_df['fau'] == 43, 'intensity'] = 1
                # cur_pspi_seq[cur_time,0] = cur_df_lbl
                cur_np   = np.hstack((np.ones((cur_df.shape[0],1),dtype=int)*cur_time, cur_df.to_numpy()))
                lst_rcv  = np.vstack(( lst_rcv, cur_np ))

        lst_rcvFinal = lst_rcv + 0
        lst_rcvFinal[:, 1] = replace_with_dict(lst_rcv[:,1], dic)
        cur_facs_seq = coo_matrix( ( lst_rcvFinal[:, 2], (lst_rcvFinal[:, 0], lst_rcvFinal[:, 1]) ),
                                   shape=(len(file_list), 14) ).toarray()
        cur_facs_seq = np.hstack(( np.expand_dims(np.arange(len(file_list)),1) ,cur_facs_seq ))
        # np.savetxt(os.path.join(out_pspi_dir, cur_seq + ".csv"), cur_pspi_seq, delimiter=',', fmt='%d', )
        np.savetxt(os.path.join(out_facs_dir, cur_seq + ".csv"), cur_facs_seq, delimiter=',', fmt='%d',)
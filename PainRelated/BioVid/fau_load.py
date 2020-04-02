import glob
import os
import pandas as pd
import numpy as np
from ccc import compute_ccc
from ccc import ccc_loss_1, ccc_loss_2, ccc_loss_3
from keras import regularizers
from keras.models import load_model
from keras.layers import Input, Dense, Activation, TimeDistributed, Bidirectional, Dropout, CuDNNLSTM, BatchNormalization, Conv1D
from zipfile import ZipFile

def get_bl1_mapping():
    # BL1 mapping import
    mytxt1="/home/vedhas/workspace/EmoPain/BioVid/PartA_ACII2017_Paper/all_BL1_orig.txt"
    mytxt2="/home/vedhas/workspace/EmoPain/BioVid/PartA_ACII2017_Paper/all_BL2.txt"

    ls1 = pd.read_csv(mytxt1,sep='\t',header=None)[0].to_list()     #renamed
    ls2 = pd.read_csv(mytxt2,sep='\t',header=None)[0].to_list()     #original

    ls1 = [cur_name[2:22].replace('-','',1) for cur_name in ls1]
    BL1_dict = dict(zip(ls2,ls1))
    return BL1_dict


def select435manipulations():    
    # The 87 expressiveness labels from the manually annotated 435 sequences
    select435path = "/home/vedhas/workspace/EmoPain/BioVid/PartA_ACII2017_Paper/Selected435.txt"
    with open(select435path, 'r') as f:
        select435_file_names = sorted([line.strip() for line in f])
    select435_biovid_pspi4 = np.array([3, 6, 2, 3, 3, 5, 3, 3, 1, 3, 2, 3, 2, 7, 4, 1, 0, 8, 1, 7, 5, 0, 0, 5, 2, 2, 0, 1, 3, 0, 3, 0, 0, 1, 1, 0, 4, 3, 0, 0, 0, 0, 3, 0, 2, 3, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 2, 1, 0, 5, 0, 1, 2, 3, 3, 0, 1, 1, 1, 0, 3, 1, 2, 2, 5, 4, 3, 2, 4, 0, 4, 3, 1, 4, 1])
    select435_biovid_pspisd = np.array([1.41421, 3.27109, 1.09545, 1.58114, 1.41421, 2.07364, 1.30384, 1.34164, 0.54772, 1.34164, 0.89443, 1.87083, 1.34164, 3.04959, 2.28035, 0.54772, 0.00000, 4.38178, 0.89443, 2.94958, 2.19089, 0.00000, 0.00000, 2.30217, 1.78885, 1.09545, 0.00000, 0.44721, 1.34164, 0.00000, 1.34164, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.78885, 1.22474, 0.44721, 0.00000, 0.00000, 0.44721, 1.30384, 0.44721, 0.89443, 3.27109, 1.30384, 0.83666, 0.00000, 0.00000, 0.54772, 0.54772, 0.00000, 2.16795, 0.00000, 0.00000, 0.44721, 0.00000, 1.34164, 2.58844, 0.00000, 1.87083, 0.00000, 0.54772, 0.70711, 1.34164, 1.22474, 0.00000, 0.00000, 0.00000, 0.54772, 0.44721, 1.22474, 0.00000, 0.83666, 1.09545, 2.07364, 2.04939, 1.34164, 0.89443, 1.78885, 0.00000, 1.64317, 1.64317, 0.00000, 2.07364, 1.00000])
    select435_subjects_formatted = [cur_name[2:14].replace('-','',1) for cur_name in select435_file_names]
    select435_files_formatted = [cur_name[2:22].replace('-','',1)  for cur_name in select435_file_names]

    return select435_file_names, select435_files_formatted, select435_subjects_formatted

def bio_to_video_mapping():

    fau_pattern = "/home/vedhas/workspace/EmoPain/BioVid/partA/FAU/*.csv"
    fau_path_list = sorted(glob.glob(fau_pattern))
    fau_names = sorted([os.path.basename(cur_path) for cur_path in fau_path_list])

    bio_zip_path='/home/vedhas/workspace/EmoPain/BioVid/partA/biosignals_filtered.zip'
    zip_file = ZipFile(bio_zip_path)
    '''    
    dfs = {text_file.filename: pd.read_csv(zip_file.open(text_file.filename),sep='\t')
       for text_file in zip_file.infolist()
       if text_file.filename.endswith('.csv')}
    '''    
    bio_path_list =sorted([text_file.filename for text_file in zip_file.infolist() if text_file.filename.endswith('.csv')])
    bio_names = sorted([cur_path[32:] for cur_path in bio_path_list])
    
    df = pd.DataFrame({ 'original_video_name': fau_names,
                        'renamed_bio_name': bio_names,
                    })
    df.to_csv('mapping_video_bio.txt', index=False, sep='\t')
    assert(len(fau_names)==len(bio_names))
    for fau,bio in zip(fau_names,bio_names):
        print(fau, bio)        
        assert fau==bio.replace('_bio','')

def interpolate_and_sample(df,upsampleby=1):
    df.index = df['time']
    df1 = df.reindex(df.index.union(np.linspace(0,5500000,138*upsampleby+1))).interpolate('index')
    df2 = df1.loc[np.linspace(0,5500000,138*upsampleby+1)].to_numpy()[1:,:]
    return df2

def bio_csv_to_numpy(upsampleby=1):
    bio_zip_path='/home/vedhas/workspace/EmoPain/BioVid/partA/biosignals_filtered.zip'
    zip_file = ZipFile(bio_zip_path)
    bio_path_list =sorted([text_file.filename for text_file in zip_file.infolist() if text_file.filename.endswith('.csv')])
    bio_names = sorted([cur_path[32:] for cur_path in bio_path_list])   #the name contains '_bio.csv'

    BL1_dict = get_bl1_mapping()
    select435_file_names, select435_files_formatted, select435_subjects_formatted = select435manipulations()

    fau_pattern = "/home/vedhas/workspace/EmoPain/BioVid/partA/FAU/*.csv"
    fau_path_list = sorted(glob.glob(fau_pattern))
    fau_names = sorted([os.path.basename(cur_path) for cur_path in fau_path_list])


    subject_pspi_dict={}
    length_list=[]
    assert len(bio_path_list)==len(fau_path_list)
    for cur_bio_path, cur_fau_path in zip(bio_path_list,fau_path_list):
        cur_name = os.path.basename(cur_bio_path).replace('.csv','').replace('_bio','')
        assert cur_name == os.path.basename(cur_fau_path).replace('.csv','')
        if 'BL1' in cur_name:  
            cur_name=BL1_dict[cur_name]
        cur_subject = cur_name[:11]
        cur_pain = cur_name[12:15]
        cur_bio_df   = pd.read_csv(zip_file.open(cur_bio_path),sep='\t')
        cur_bio_df.columns = cur_bio_df.columns.str.lstrip() #remove leading space from the column names
        # ['time', 'gsr', 'ecg', 'emg_trapezius', 'emg_corrugator','emg_zygomaticus']
        current_feats = interpolate_and_sample(cur_bio_df, upsampleby=upsampleby)

        cur_df   = pd.read_csv(cur_fau_path)
        cur_df.columns = cur_df.columns.str.lstrip() #remove leading space from the column names
        # ['frame', 'face_id', 'timestamp', 'confidence', 'success',                                    5   REMEMBER TO SUBTRACT 1 !!!
        # 'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz',                             6
        # 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r',     9
        # 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r',           'AU45_r',     8
        # 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c',     9   
        # 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c'] + now 'PSPI'     9(+1)       138rows x 47 columns
        cur_df['pspi'] = cur_df['AU04_r'] + cur_df[["AU06_r", "AU07_r"]].max(axis=1) + cur_df[["AU09_r", "AU10_r"]].max(axis=1) + cur_df['AU45_c']
        assert cur_df['AU45_c'].all()<=1 and cur_df['AU45_c'].all()>=0
        current_pspi = cur_df['pspi'].to_numpy()
        current_pspi = np.expand_dims(current_pspi,1)

        if True: #(cur_name in select435_files_formatted ):    #Whether to check against 435 filesnames, or for use all the 8700 sequences
            if cur_subject not in subject_pspi_dict:
                subject_pspi_dict[cur_subject]={}
            if cur_pain not in subject_pspi_dict[cur_subject]:
                subject_pspi_dict[cur_subject][cur_pain] = ([current_feats], [current_pspi])
            else:
                subject_pspi_dict[cur_subject][cur_pain][0].append(current_feats)             
                subject_pspi_dict[cur_subject][cur_pain][1].append(current_pspi)             
        
    bio_5d =  np.array([ [subject_pspi_dict[cur_subject][cur_pain][0] for cur_pain in ['BL1','PA1','PA2','PA3','PA4']]
                                for cur_subject in sorted(subject_pspi_dict)
                            ]) 
    pspi_5d =  np.array([ [subject_pspi_dict[cur_subject][cur_pain][1] for cur_pain in ['BL1','PA1','PA2','PA3','PA4']]
                                for cur_subject in sorted(subject_pspi_dict)
                            ]) 
    print(bio_5d.shape,pspi_5d.shape)
    np.save('bio_5d_{}.npy'.format(upsampleby) , bio_5d)
    np.save('pspi_5d.npy', pspi_5d)
    return bio_5d, pspi_5d




def fau_csv_to_numpy():

    BL1_dict = get_bl1_mapping()
    select435_file_names, select435_files_formatted, select435_subjects_formatted = select435manipulations()

    fau_pattern = "/home/vedhas/workspace/EmoPain/BioVid/partA/FAU/*.csv"
    fau_path_list = sorted(glob.glob(fau_pattern))
    subject_pspi_dict={}

    length_list=[]
    for cur_fau_path in fau_path_list:
        cur_name = os.path.basename(cur_fau_path).replace('.csv','')
        if 'BL1' in cur_name:  
            cur_name=BL1_dict[cur_name]
        cur_subject = cur_name[:11]
        cur_pain = cur_name[12:15]
        cur_df   = pd.read_csv(cur_fau_path)
        cur_df.columns = cur_df.columns.str.lstrip() #remove leading space from the column names
        # ['frame', 'face_id', 'timestamp', 'confidence', 'success',                                    5   REMEMBER TO SUBTRACT 1 !!!
        # 'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz',                             6
        # 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r',     9
        # 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r',           'AU45_r',     8
        # 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c',     9   
        # 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c'] + now 'PSPI'     9(+1)       138rows x 47 columns
        cur_df['pspi'] = cur_df['AU04_r'] + cur_df[["AU06_r", "AU07_r"]].max(axis=1) + cur_df[["AU09_r", "AU10_r"]].max(axis=1) + cur_df['AU45_c']
        assert cur_df['AU45_c'].all()<=1 and cur_df['AU45_c'].all()>=0
        current_pspi = cur_df['pspi'].to_numpy()
        current_pspi = np.expand_dims(current_pspi,1)
        current_feats = cur_df.iloc[:,np.arange(3,46)].to_numpy()
        length_list.append(len(current_pspi))  
        if True: #(cur_name in select435_files_formatted ):    #Whether to check against 435 filesnames, or for use all the 8700 sequences
            if cur_subject not in subject_pspi_dict:
                subject_pspi_dict[cur_subject]={}
            if cur_pain not in subject_pspi_dict[cur_subject]:
                subject_pspi_dict[cur_subject][cur_pain] = ([current_feats], [current_pspi])
            else:
                subject_pspi_dict[cur_subject][cur_pain][0].append(current_feats)             
                subject_pspi_dict[cur_subject][cur_pain][1].append(current_pspi)             
            '''
            if 'PA4' in cur_name:
                assert len(subject_pspi_dict[cur_subject])==5
                computed_pspi_pa4_max.append(np.max(current_pspi))
                computed_pspi_pa4_std.append(np.std(current_pspi))
                computed_pspi_pa4_mean.append(np.mean(current_pspi))
                computed_pspi_pa4_median.append(np.median(current_pspi))
                computed_pspi_sd_max.append(float('{:.6}'.format(np.std([np.max(cur_pspi_list) for cur_pspi_list in subject_pspi_dict[cur_subject]],ddof=0))))
                computed_pspi_max_sd.append(float('{:.6}'.format(np.max([np.std(cur_pspi_list,ddof=0) for cur_pspi_list in subject_pspi_dict[cur_subject]]))))
                computed_pspi_sd_median.append(float('{:.6}'.format(np.std([np.median(cur_pspi_list) for cur_pspi_list in subject_pspi_dict[cur_subject]],ddof=0))))
                computed_pspi_sd_mean.append(float('{:.6}'.format(np.std([np.mean(cur_pspi_list) for cur_pspi_list in subject_pspi_dict[cur_subject]],ddof=0))))
            '''
    '''        
    for cur_subject in subject_pspi_dict:
        assert len(subject_pspi_dict[cur_subject])==5
        for cur_pain in subject_pspi_dict[cur_subject]:
            assert len(subject_pspi_dict[cur_subject][cur_pain])==2
            assert len(subject_pspi_dict[cur_subject][cur_pain][0])==20
            assert len(subject_pspi_dict[cur_subject][cur_pain][1])==20
    '''
    fau_5d =  np.array([ [subject_pspi_dict[cur_subject][cur_pain][0] for cur_pain in ['BL1','PA1','PA2','PA3','PA4']]
                                for cur_subject in sorted(subject_pspi_dict)
                            ]) 
    pspi_5d =  np.array([ [subject_pspi_dict[cur_subject][cur_pain][1] for cur_pain in ['BL1','PA1','PA2','PA3','PA4']]
                                for cur_subject in sorted(subject_pspi_dict)
                            ]) 
    print(fau_5d.shape,pspi_5d.shape)
    np.save('fau_5d.npy' , fau_5d)
    np.save('pspi_5d.npy', pspi_5d)
    return fau_5d, pspi_5d

def generate_model_cnn(max_seq_len, num_features, num_targets, num_cells_1, num_cells_2, num_cells_3, num_cells_4, batch_norm, last_specific, final_activation, dropout=0.0, stride=1):
    # Input
    inputs = Input(shape=(max_seq_len ,num_features))
    net = inputs

    if num_cells_1[0] > 0:
        # 1st layer
        net = Conv1D(num_cells_1[0], num_cells_1[1], strides=1, padding='same')(net)
        if batch_norm: net = BatchNormalization()(net)
        net = Activation('linear')(net)
        net = Dropout(rate=dropout)(net)

    # 2nd layer
    if num_cells_2[0] > 0:
        net = Conv1D(num_cells_2[0], num_cells_2[1], strides=1, padding='same')(net)
        if batch_norm: net = BatchNormalization()(net)
        net = Activation('linear')(net)
        net = Dropout(rate=dropout)(net)

    # 3rd layer
    if num_cells_3[0] > 0:
        net = Conv1D(num_cells_3[0], num_cells_3[1], strides=1, padding='same')(net)
        if batch_norm: net = BatchNormalization()(net)
        net = Activation('linear')(net)
        net = Dropout(rate=dropout)(net)

    if not last_specific:
        # 4th layer
        if num_cells_4[0] > 0:
            net = Conv1D(num_cells_4[0], num_cells_4[1], strides=stride, padding='same', 
                        kernel_regularizer=regularizers.l1(0.01), 
                        #bias_regularizer=regularizers.l1(0.001)
                    )(net)
            if batch_norm: net = BatchNormalization()(net)
            #net = Activation(final_activation)(net)
            #net = Dropout(rate=dropout)(net)

        # outputs (& task-specific layers)
        out = []
        for n in range(num_targets):
            outn = TimeDistributed(Dense(1))(net)
            #outn = Activation(final_activation)(outn)
            out.append(outn)
    else:  # 4th layer mandatory!
        out = []
        for n in range(num_targets):
            net_part = Conv1D(num_cells_4[0], num_cells_4[1], strides=1, padding='same')(net)
            if batch_norm: net_part = BatchNormalization()(net_part)
            net_part = Activation('linear')(net_part)
            net_part = Dropout(rate=dropout)(net_part)
            #
            outn = TimeDistributed(Dense(1))(net_part)
            outn = Activation(final_activation)(outn)
            out.append(outn)

    return inputs, out

def get_loss(loss_function):
    if   loss_function=='ccc_1': loss = ccc_loss_1
    elif loss_function=='ccc_2': loss = ccc_loss_2  # not faster, maybe(!) better in terms of the result
    elif loss_function=='ccc_3': loss = ccc_loss_3
    elif loss_function=='mse':   loss = 'mean_squared_error'
    return loss


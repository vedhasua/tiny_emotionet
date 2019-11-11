from deepemotion_keras import main

param_given      = True      #f
culture          = 'German'  #f
eval_cross       = True      #f
modality         = 'audio'   #f
get_turn_feature = True      #f

uncertainty_target = False   #f
invert_std         = False   #f
loss_unc           = 'ccc_2' #f
weight_unc         = 0.5     #f

balance_weight     = False  #f
uncertainty_weight = False  #f

batch_sizes      = [34]     #PARAM 1 better for valence, 34 better for arousal
learning_rates   = [0.001]  #[0.00025,0.0005,0.001,0.002]  ## PARAM
#max_num_epochs   = 500     #f 500/100 (for BS 1)

first_lstm       = True     #f
num_cells_1      = 200      #f 
num_cells_2      = 64       #f 
num_cells_3      = 32       #f 
num_cells_4      = 32       #f 
#num_cells_1      = [200,5]  #f 
#num_cells_2      = [64,20]  #f 
#num_cells_3      = [32,30]  #f 
#num_cells_4      = [32,50]  #f 
last_lstm        = False    #f
batch_norm       = False    #f - requires a high learning rate, but no improvement
last_specific    = False    #f - no multi-task for the beginning

comb_smoothing   = False    #na
bidirectional    = True     #na
dropout          = 0.0      #f - no big difference
final_activation = 'linear' #f - tanh does not work for CNN
loss_function    = 'ccc_2'  #f
shift_secs       = [0.0,0.4,0.8,1.2,1.6,2.0,2.4,2.8,3.2,3.6,4.0,4.4,4.8,5.2,5.6,6.0]  #PARAM - 0.05 opt for window size 0.1, uni-directional LSTM

targets_avls   = ['A','V']  #f
feature_type_a = 'funcegem' #f 'mfcc' & 'funcegem' best, 'egem' works best for fusion, 'mfcccomp' worse, 'funccomp' better for valence, but bad for arousal on devel
feature_type_v = 'faus'     #f 'faus+lips' have approx. the same performance
window_size    = 0.5        #f
xbow_cs        = 1000       #na
xbow_na        = 10         #na
random_seeds   = [0]        ## PARAM
add_noise      = False      #f # not implemented

append_results_file = 'all_results_lstm.txt'

for targets_avl in targets_avls:
  for shift_sec in shift_secs:
    for batch_size in batch_sizes:
        ##
        if batch_size==1:
          max_num_epochs = 100
        elif batch_size<10:
          max_num_epochs = 250
        else:
          max_num_epochs = 200
        ##
        for learning_rate in learning_rates:
          for random_seed in random_seeds:
            main(param_given,
                 culture=culture,
                 eval_cross=eval_cross,
                 modality=modality,
                 get_turn_feature=get_turn_feature,
                 uncertainty_target=uncertainty_target,
                 invert_std=invert_std,
                 loss_unc=loss_unc,
                 weight_unc=weight_unc,
                 balance_weight=balance_weight,
                 uncertainty_weight=uncertainty_weight,
                 batch_size=batch_size,
                 learning_rate=learning_rate,
                 max_num_epochs=max_num_epochs,
                 first_lstm=first_lstm,
                 num_cells_1=num_cells_1,
                 num_cells_2=num_cells_2,
                 num_cells_3=num_cells_3,
                 num_cells_4=num_cells_4,
                 last_lstm=last_lstm,
                 batch_norm=batch_norm,
                 last_specific=last_specific,
                 comb_smoothing=comb_smoothing,
                 bidirectional=bidirectional,
                 dropout=dropout,
                 final_activation=final_activation,
                 loss_function=loss_function,
                 shift_sec=shift_sec,
                 targets_avl=targets_avl,
                 feature_type_a=feature_type_a,
                 feature_type_v=feature_type_v,
                 window_size=window_size,
                 xbow_cs=xbow_cs,
                 xbow_na=xbow_na,
                 random_seed=random_seed,
                 add_noise=add_noise,
                 append_results_file=append_results_file)


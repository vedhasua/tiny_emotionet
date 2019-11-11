import numpy as np
from ccc import compute_ccc
from scipy.signal import medfilt


def median_filter(pred, kernel, copy=True):
    if kernel%2==0:  # scipy.signal.medfilt does not accept even numbers
        kernel += 1
    if copy:
        pred_new = []
        for seq in pred:
            pred_new.append(medfilt(seq, kernel))
        pred = pred_new
    else:
        for k in range(0, len(pred)):
            pred[k]   = medfilt(pred[k], kernel)
    return pred

def flatten(pred):
    pred_flat = np.empty(0)
    for seq in pred:
        pred_flat = np.concatenate((pred_flat, seq))
    return pred_flat

def train(gold, pred, Nw=100, wstep=4, Nshift=200):
    # gold/pred: list of numpy arrays, each array has 1 dimension (sequence_length,)
    # e.g., gold = [np.array([.5,.3,.6,.4,.7,.3]), np.array([-.1,.2,-.2,.1])] for two sequences of length 6 and 4
    # Nw=100:     adapted for a hop size of 0.1s
    # wstep=4:    adapted for a hop size of 0.1s
    # Nshift=200: adapted for a hop size of 0.1s
    
    # raw, filter, center, scale, shift
    CCC_save = np.zeros(5)
    # list: wmedian, bias, scale, shift
    best_param = []
    
    # flatten gold standard and predictions
    gold_flat = flatten(gold)
    
    # compute performance on raw
    CCC_save[0] = compute_ccc(gold_flat, flatten(pred))
    best_CCC    = CCC_save[0]
    
    # filter by dichotomy (makes computation faster ...)
    perc_impr = 2.
    
    new_val    = np.zeros(4)
    new_val[0] = CCC_save[0]
    
    for k in range(0,3):
        kernel       = (k+1)*Nw/3
        pred_filt    = median_filter(pred, kernel)
        new_val[k+1] = compute_ccc(gold_flat, flatten(pred_filt))
    
    val  = np.flip(np.sort(new_val),    axis=0)
    indw = np.flip(np.argsort(new_val), axis=0)
    
    if indw[0]!=0 and 100*(val[0]-new_val[0])/new_val[0] > perc_impr:
        # filtering useful - perform second round of dichotomy
        indw      = indw[0:2]*Nw/3
        new_indw  = [indw[0], int(np.round(np.mean(indw))), indw[1]]
        kernel    = new_indw[1]
        pred_filt = median_filter(pred, kernel)
        new_val   = [val[0], compute_ccc(gold_flat, flatten(pred_filt)), val[1]]
        
        # continue dichotomy if still improvement
        if 100*(new_val[1]-max(new_val[0],new_val[2]))/max(new_val[0],new_val[2]) > perc_impr:
            eot = True
            while eot:
                val       = np.flip(np.sort(new_val),    axis=0)
                indw      = np.flip(np.argsort(new_val), axis=0)
                new_indw  = [new_indw[indw[0]], int(np.round(np.mean([new_indw[indw[0]],new_indw[indw[1]]]))), new_indw[indw[1]]]
                kernel    = new_indw[1]
                pred_filt = median_filter(pred, kernel)
                new_val   = [val[0], compute_ccc(gold_flat, flatten(pred_filt)), val[1]]
                
                # if there is yet improvement
                if 100*(new_val[1]-max(new_val[0],new_val[2]))/max(new_val[0],new_val[2]) > 0.:
                    if len(best_param)==0:  # TODO: not so nice
                        best_param.append(new_indw[1])
                    else:  # TODO
                        best_param[0]=new_indw[1]
                    best_CCC = new_val[1]
                else:
                    if len(best_param)==0:  # TODO: not so nice
                        best_param.append(new_indw[0])
                    else:  # TODO
                        best_param[0]=new_indw[0]
                    best_CCC = val[0]
                    eot = False
        else:
            if 100*(new_val[1]-max(new_val[0],new_val[2]))/max(new_val[0],new_val[2]) > 0.:
                best_param.append(new_indw[1])
                best_CCC = new_val[1]
            else:
                best_param.append(indw[0])
                best_CCC = val[0]
    else:
        best_param.append(0)
    
    # apply median filtering
    if best_param[-1] > 1:
        pred = median_filter(pred, best_param[-1], copy=False)
    CCC_save[1] = best_CCC
    
    
    ## center prediction
    pred_flat   = flatten(pred)  # Flatten latest filtered predictions
    mean_gold   = np.mean(gold_flat)
    mean_pred   = np.mean(pred_flat)
    bias        = mean_gold - mean_pred
    pred_center = pred_flat + bias
    
    CCC_tmp = compute_ccc(gold_flat, pred_center)
    # save configuration if improvement
    if CCC_tmp > best_CCC:
        best_param.append(bias)
        best_CCC = CCC_tmp
        # Apply bias to all sequences
        for ind in range(0, len(pred)):
            pred[ind] = pred[ind] + bias
    else:
        best_param.append(0)
    CCC_save[2] = best_CCC
    

    ## scale prediction
    pred_flat  = flatten(pred)  # Flatten latest filtered predictions
    std_gold   = np.std(gold_flat)
    std_pred   = np.std(pred_flat)
    scale      = std_gold / std_pred
    pred_scale = pred_flat * scale
    
    # save configuration if improvement
    CCC_tmp = compute_ccc(gold_flat, pred_scale)
    if CCC_tmp > best_CCC:
        best_param.append(scale)
        best_CCC = CCC_tmp
        # Apply scaling to all sequences
        for ind in range(0, len(pred)):
            pred[ind] = pred[ind] * scale
    else:
        best_param.append(1)
    CCC_save[3] = best_CCC
    
    
    ## shift prediction backward / forward
    CCC_tmp               = np.zeros(2*Nshift/wstep+1)
    CCC_tmp[Nshift/wstep] = best_CCC
    for shift in range(1, Nshift/wstep+1):
        tmp_flat = np.empty(0)
        for seq in pred:
            tmp      = np.concatenate((seq[shift*wstep:], np.repeat(seq[-1], shift*wstep)))
            tmp_flat = np.concatenate((tmp_flat, tmp))
        CCC_tmp[Nshift/wstep-shift] = compute_ccc( gold_flat, tmp_flat )
    for shift in range(1, Nshift/wstep+1):
        tmp_flat = np.empty(0)
        for seq in pred:
            tmp      = np.concatenate((np.repeat(seq[0], shift*wstep), seq[:-shift*wstep]))
            tmp_flat = np.concatenate((tmp_flat, tmp))
        CCC_tmp[Nshift/wstep+shift] = compute_ccc( gold_flat, tmp_flat )
    val = np.max(CCC_tmp)
    ind = np.argmax(CCC_tmp)
    
    # save configuration if improvement
    if val > best_CCC:
        shift_optim = (ind-Nshift/wstep) * wstep
        best_param.append(shift_optim)
        best_CCC = val
        
        if shift_optim > 0:
            for k in range(0, len(pred)):
                pred[k] = np.concatenate(( np.repeat(pred[k][0], shift_optim), pred[k][:-shift_optim] ))
        elif shift_optim < 0:
            shift_optim = -shift_optim
            for k in range(0, len(pred)):
                pred[k] = np.concatenate(( pred[k][shift_optim:], np.repeat(pred[k][-1], shift_optim) )) 
    else:
        best_param.append(0)
    CCC_save[4] = best_CCC
    
    return CCC_save, best_param  # pred is filtered implicitly


def predict(gold, pred, best_param):
    # gold/pred: list of numpy arrays, each array has 1 dimension (sequence_length,)
    # e.g., gold = [np.array([.5,.3,.6,.4,.7,.3]), np.array([-.1,.2,-.2,.1])] for two sequences of length 6 and 4
    # best_param: [wmedian, bias, scale, shift] - obtained by train
    
    # raw, filter, center, scale, shift
    CCC_save = np.zeros(5)
    
    # 
    gold_flat = flatten(gold)
    
    # compute performance on raw
    CCC_save[0] = compute_ccc(gold_flat, flatten(pred))
    
    # apply filtering
    if best_param[0] > 0:
        kernel = best_param[0]
        pred   = median_filter(pred, kernel, copy=False)
        CCC_save[1] = compute_ccc(gold_flat, flatten(pred))
    else:
        CCC_save[1] = CCC_save[0]
    
    # apply centering
    if best_param[1] != 0:
        for k in range(0, len(pred)):
            pred[k] = pred[k] + best_param[1]
        CCC_save[2] = compute_ccc(gold_flat, flatten(pred))
    else:
        CCC_save[2] = CCC_save[1]
    
    # apply scaling
    if best_param[2] != 1:
        for k in range(0, len(pred)):
            pred[k] = pred[k] * best_param[2]
        CCC_save[3] = compute_ccc(gold_flat, flatten(pred))
    else:
        CCC_save[3] = CCC_save[2]
    
    # apply shifting
    shift_optim = best_param[3]
    if shift_optim > 0:
        for k in range(0, len(pred)):
            pred[k] = np.concatenate(( np.repeat(pred[k][0], shift_optim), pred[k][:-shift_optim] ))
        CCC_save[4] = compute_ccc(gold_flat, flatten(pred))
    elif shift_optim < 0:
        shift_optim = -shift_optim
        for k in range(0, len(pred)):
            pred[k] = np.concatenate(( pred[k][shift_optim:], np.repeat(pred[k][-1], shift_optim) )) 
        CCC_save[4] = compute_ccc(gold_flat, flatten(pred))
    else:
        CCC_save[4] = CCC_save[3]
    
    return CCC_save  # pred is filtered implicitly

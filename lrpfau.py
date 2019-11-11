import innvestigate
import imp
import os
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow as tf
import numpy as np
from ccc import ccc_loss_1, ccc_loss_2, ccc_loss_3
from keras.models import Model, load_model, model_from_json
from modelconversion import convert_to_2d, convert_to_2dpure

from matplotlib import pyplot as plt
import innvestigate
import innvestigate.utils as iutils
import innvestigate.applications.imagenet
from ccc import compute_ccc

def shift_annotations_to_front(labels, shift=0):
    labels = np.concatenate((labels[:,shift:,:], np.zeros((labels.shape[0],shift,labels.shape[2]))), axis=1)
    return labels

def shift_annotations_to_back(labels, shift=0):
    print(labels.shape)
    labels = np.concatenate((np.zeros((labels.shape[0],shift,labels.shape[2])), labels[:,:labels.shape[1]-shift,:]), axis=1)
    return labels
# Use utility libraries to focus on relevant iNNvestigate routines.
# eutils = imp.load_source("utils", "/home/vedhas/Setups/innvestigate/examples/utils.py")
# mnistutils = imp.load_source("utils_mnist", "/home/vedhas/Setups/innvestigate/examples/utils_mnist.py")
imgnetutils = imp.load_source("utils_imagenet", "/home/vedhas/Setups/innvestigate/examples/utils_imagenet.py")

# Methods we use and some properties.
'''
methods = [
    # NAME                    OPT.PARAMS                POSTPROC FXN                TITLE
    # Show input.
    ("input",                 {},                       imgnetutils.image,         "Input"),

    # Function
    ("gradient",              {"postprocess": "abs"},   imgnetutils.graymap,       "Gradient"),
    ("smoothgrad",            {"augment_by_n": 64,
                               "noise_scale": noise_scale,
                               "postprocess": "square"},imgnetutils.graymap,       "SmoothGrad"),

    # Signal
    ("deconvnet",             {},                       imgnetutils.bk_proj,       "Deconvnet"),
    ("guided_backprop",       {},                       imgnetutils.bk_proj,       "Guided Backprop",),
    ("pattern.net",           {"patterns": patterns},   imgnetutils.bk_proj,       "PatternNet"),

    # Interaction
    ("pattern.attribution",   {"patterns": patterns},   imgnetutils.heatmap,       "PatternAttribution"),
    ("deep_taylor.bounded",   {"low": input_range[0],
                               "high": input_range[1]}, imgnetutils.heatmap,       "DeepTaylor"),
    ("input_t_gradient",      {},                       imgnetutils.heatmap,       "Input * Gradient"),
    ("integrated_gradients",  {"reference_inputs": input_range[0],
                               "steps": 64},            imgnetutils.heatmap,       "Integrated Gradients"),
    ("lrp.z",                 {},                       imgnetutils.heatmap,       "LRP-Z"),
    ("lrp.epsilon",           {"epsilon": 1},           imgnetutils.heatmap,       "LRP-Epsilon"),
    ("lrp.sequential_preset_a_flat",{"epsilon": 1},     imgnetutils.heatmap,       "LRP-PresetAFlat"),
    ("lrp.sequential_preset_b_flat",{"epsilon": 1},     imgnetutils.heatmap,       "LRP-PresetBFlat"),
]
'''
# tmp = getattr(innvestigate.applications.imagenet, os.environ.get("NETWORKNAME", "vgg16"))
# net = tmp(load_weights=True, load_patterns="relu")
# patterns = net["patterns"]
# input_range = net["input_range"]
# noise_scale = (input_range[1]-input_range[0]) * 0.1
#
# curMethod=("pattern.attribution",   {"patterns": patterns},   imgnetutils.heatmap,       "PatternAttribution"),

modelFile          = '/home/vedhas/workspace/is2019_recurrence/DHCresults/2/m1000_tr0.687_dv0.608_ts0.625_tc0.420.h5'
model = load_model(modelFile, custom_objects={'ccc_loss_2': ccc_loss_2})
model2 = convert_to_2dpure(model)
feature_type_v='faus'
targets_avl='V'
fsuffix=targets_avl+"_v_{}".format(feature_type_v)

DeNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/De{}.npy'.format(fsuffix)
HuNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/Hu{}.npy'.format(fsuffix)
CnNPY            = '/home/vedhas/workspace/is2019_recurrence/NPY/Cn{}.npy'.format(fsuffix)

if os.path.isfile(DeNPY):
    trainX, trainY, develX, develY, testX, testY, origTrain, origDevel, origTest = np.load(DeNPY,allow_pickle=True, encoding = 'bytes')
if os.path.isfile(CnNPY):
    trainCX, trainCY, develCX, develCY, testCX, testCY, origCTrain, origCDevel, origCTest, origCrossC, crossXC, crossYC = np.load(CnNPY,allow_pickle=True, encoding = 'bytes')
if os.path.isfile(HuNPY):
    trainHX, trainHY, develHX, develHY, testHX, testHY, origHTrain, origHDevel, origHTest, origCross, crossX, crossY = np.load(HuNPY,allow_pickle=True, encoding = 'bytes')

# innvestigate.analyzer.pattern_based.PatternNet(model2)\
# pattern.attribution
# testCX = np.expand_dims(testCX[0:1,:],-1)

out1 = model.predict(testCX)
out2 = model2.predict(np.expand_dims(testCX, axis=-1))[:,:,0,:] #[:,:,18,:]
out3 = shift_annotations_to_back(out2,28)
'''
for curSubId in range(testCY[0].shape[0]):
    print(compute_ccc(out2[curSubId].flatten(), testCY[0][curSubId].flatten()),\
          compute_ccc(out3[curSubId].flatten(), testCY[0][curSubId].flatten()))
'''

plt.plot(out1[34].flatten(),'b')
plt.plot(out2[34].flatten(),'r*')
plt.plot(testCY[0][34].flatten())
plt.plot(testCX[34,:,10].flatten())
plt.show()

# testCX = imgnetutils.preprocess(testCX, model2)
# pattern.attribution
testCX_expanded=np.expand_dims(testCX, axis=-1)
print(testCX_expanded.shape)
analyzer = innvestigate.create_analyzer('input_t_gradient', model2, neuron_selection_mode='index')
# analyzer.fit(testCX_expanded)
a = analyzer.analyze(testCX_expanded,1)
print('postanalysis: a =',a)
a = imgnetutils.postprocess(a,None,False)
print('postprocess: a =',a)
a= imgnetutils.heatmap(a)
print('postheatmap, a[0] = ',a[0])
# analyzer = innvestigate.create_analyzer(curMethod,        # analysis method identifier
#                                         model_wo_softmax, # model without softmax output
#                                         neuron_selection_mode="index",
#                                         **method[1])      # optional analysis parameters
# analyzer.fit(np.expand_dims(trainX,-1), batch_size=256, verbose=1)
# a = analyzer.analyze(np.expand_dims(testCX,-1))
# print(a)


'''
# https://github.com/albermax/innvestigate/issues/113
from keras import Sequential
from keras.layers import Dense, Conv1D, Embedding, GlobalMaxPooling1D
import numpy as np
import innvestigate

model = Sequential()
model.add(Embedding(input_dim=219, output_dim=8))
model.add(Conv1D(filters=64, kernel_size=8, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation=None))

#test
model.predict(np.random.randint(1,219, (1,100)))  # [[0.04913538 0.04234646]]

analyzer = innvestigate.create_analyzer('lrp.epsilon', model, neuron_selection_mode="max_activation", **{'epsilon': 1})
analyzer = innvestigate.create_analyzer('input_t_gradient', model)
analyzer.analyze(np.random.randint(1, 219, (1,100)))
'''

import numpy as np


def standardise_3(train, devel, test):
  max_seq_len  = train.shape[1]
  num_features = train.shape[2]
  # Estimate mean and stddev only from train
  MEAN   = np.mean(train.reshape([-1,num_features]),0)
  STDDEV = np.std(train.reshape([-1,num_features]),0)
  # Standardise train and test with the same parameters
  train = train.reshape([-1,num_features]) - MEAN
  train = train / (STDDEV + np.finfo(np.float32).eps)
  train = train.reshape([-1,max_seq_len,num_features])
  devel = devel.reshape([-1,num_features]) - MEAN
  devel = devel / (STDDEV + np.finfo(np.float32).eps)
  devel = devel.reshape([-1,max_seq_len,num_features])
  test  = test.reshape([-1,num_features]) - MEAN
  test  = test / (STDDEV + np.finfo(np.float32).eps)
  test  = test.reshape([-1,max_seq_len,num_features])
  return train, devel, test


def standardise_3_masked(train, devel, test):  
  # values at the end masked by 0. are not considered for standardisation
  max_seq_len   = train.shape[1]
  num_features  = train.shape[2]
  
  # Estimate parameters (only from train, only where zero was not padded)
  estim         = train.reshape([-1,num_features])
  estim_max_abs = np.max(np.abs(estim), axis=1)
  mask          = np.where(estim_max_abs>0)[0]
  MEAN          = np.mean(estim[mask], axis=0)
  STDDEV        = np.std(estim[mask], axis=0)
  
  # Standardise train and test with the same parameters
  for inst in range(0, train.shape[0]):
    train_max_abs = np.max(np.abs(train[inst,:,:]), axis=1)
    mask          = np.where(train_max_abs>0)[0]
    train[inst,mask,:] = train[inst,mask,:] - MEAN
    train[inst,mask,:] = train[inst,mask,:] / (STDDEV + np.finfo(np.float32).eps)
  for inst in range(0, devel.shape[0]):
    devel_max_abs = np.max(np.abs(devel[inst,:,:]), axis=1)
    mask          = np.where(devel_max_abs>0)[0]
    devel[inst,mask,:] = devel[inst,mask,:] - MEAN
    devel[inst,mask,:] = devel[inst,mask,:] / (STDDEV + np.finfo(np.float32).eps)    
  for inst in range(0, test.shape[0]):
    test_max_abs = np.max(np.abs(test[inst,:,:]), axis=1)
    mask         = np.where(test_max_abs>0)[0]
    test[inst,mask,:] = test[inst,mask,:] - MEAN
    test[inst,mask,:] = test[inst,mask,:] / (STDDEV + np.finfo(np.float32).eps)    
  
  return train, devel, test


def standardise_2(train, test):
  max_seq_len  = train.shape[1]
  num_features = train.shape[2]
  # Estimate mean and stddev only from train
  MEAN   = np.mean(train.reshape([-1,num_features]),0)
  STDDEV = np.std(train.reshape([-1,num_features]),0)
  # Standardise train and test with the same parameters
  train = train.reshape([-1,num_features]) - MEAN
  train = train / (STDDEV + np.finfo(np.float32).eps)
  train = train.reshape([-1,max_seq_len,num_features])
  test  = test.reshape([-1,num_features]) - MEAN
  test  = test / (STDDEV + np.finfo(np.float32).eps)
  test  = test.reshape([-1,max_seq_len,num_features])
  return train, test




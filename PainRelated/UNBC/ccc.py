import numpy as np
import tensorflow as tf
import keras.backend as K


def compute_ccc(prediction,gold_standard):
    g_mean = np.nanmean(gold_standard)
    p_mean = np.nanmean(prediction)
    covariance = np.nanmean((gold_standard-g_mean)*(prediction-p_mean))
    g_var = 1.0 / (len(gold_standard)-1) * np.nansum((gold_standard-g_mean)**2)  # Make it consistent with Matlab's nanvar (division by len(x)-1, not len(x)))
    p_var = 1.0 / (len(prediction)-1) * np.nansum((prediction-p_mean)**2)
    CCC = (2*covariance) / (g_var + p_var + (g_mean-p_mean)**2)
    return CCC


def ccc_loss_old(gold_standard,prediction):
  # TODO: Old version - not correct as only the mean is returned
  g_mean = tf.reduce_mean(gold_standard)
  p_mean = tf.reduce_mean(prediction)
  covariance = tf.reduce_mean((gold_standard-g_mean)*(prediction-p_mean))
  x_var = tf.truediv(1.0,tf.cast(tf.size(gold_standard)-1,dtype=tf.float32)) * tf.reduce_sum(tf.square(gold_standard-g_mean))  # Make it consistent with Matlab's nanvar (division by len(x)-1, not len(x)))
  y_var = tf.truediv(1.0,tf.cast(tf.size(prediction)-1,dtype=tf.float32))    * tf.reduce_sum(tf.square(prediction-p_mean))
  CCC   = tf.truediv(2*covariance, x_var + y_var + tf.square(g_mean-p_mean))
  loss  = 1 - CCC
  return loss


def pcc_loss(seq1, seq2):  # Pearson's correlation coefficient (PCC)-based loss function
    # input (num_batches, seq_len, 1)
    seq1        = K.squeeze(seq1, axis=-1)  # To remove the last dimension
    seq2        = K.squeeze(seq2, axis=-1)  # To remove the last dimension
    seq1_mean   = K.mean(seq1, axis=-1, keepdims=True)
    seq2_mean   = K.mean(seq2, axis=-1, keepdims=True)
    nominator   = (seq1-seq1_mean) * (seq2-seq2_mean)
    denominator = K.sqrt( K.mean(K.square(seq1-seq1_mean), axis=-1, keepdims=True) * K.mean(K.square(seq2-seq2_mean), axis=-1, keepdims=True) )
    corr        = nominator / (denominator + K.common.epsilon())
    corr_loss   = K.constant(1.) - corr
    return corr_loss


def ccc_loss_1(seq1, seq2):  # Concordance correlation coefficient (CCC)-based loss function - using inductive statistics
    # input (num_batches, seq_len, 1)
    seq1       = K.squeeze(seq1, axis=-1)
    seq2       = K.squeeze(seq2, axis=-1)
    seq1_mean  = K.mean(seq1, axis=-1, keepdims=True)
    seq2_mean  = K.mean(seq2, axis=-1, keepdims=True)
    cov        = (seq1-seq1_mean)*(seq2-seq2_mean)
    seq1_var   = K.constant(1.) / (K.cast(K.shape(seq1)[-1], dtype='float32') - K.constant(1.)) * K.sum(K.square(seq1-seq1_mean), axis=-1, keepdims=True)
    seq2_var   = K.constant(1.) / (K.cast(K.shape(seq2)[-1], dtype='float32') - K.constant(1.)) * K.sum(K.square(seq2-seq2_mean), axis=-1, keepdims=True)
    CCC        = K.constant(2.) * cov / (seq1_var + seq2_var + K.square(seq1_mean - seq2_mean) + K.common.epsilon())
    CCC_loss   = K.constant(1.) - CCC
    return CCC_loss


def ccc_loss_2(seq1, seq2):  # Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
    # input (num_batches, seq_len, 1)
    seq1       = K.squeeze(seq1, axis=-1)
    seq2       = K.squeeze(seq2, axis=-1)
    seq1_mean  = K.mean(seq1, axis=-1, keepdims=True)
    seq2_mean  = K.mean(seq2, axis=-1, keepdims=True)
    cov        = (seq1-seq1_mean)*(seq2-seq2_mean)
    seq1_var   = K.mean(K.square(seq1-seq1_mean), axis=-1, keepdims=True)
    seq2_var   = K.mean(K.square(seq2-seq2_mean), axis=-1, keepdims=True)
    CCC        = K.constant(2.) * cov / (seq1_var + seq2_var + K.square(seq1_mean - seq2_mean) + K.common.epsilon())
    CCC_loss   = K.constant(1.) - CCC
    return CCC_loss


def ccc_loss_3(seq1, seq2):  # Concordance correlation coefficient (CCC)-based loss function (via PCC) - using non-inductive statistics
    seq1        = K.squeeze(seq1, axis=-1)
    seq2        = K.squeeze(seq2, axis=-1)
    seq1_mean   = K.mean(seq1, axis=-1, keepdims=True)
    seq2_mean   = K.mean(seq2, axis=-1, keepdims=True)
    nominator   = (seq1-seq1_mean) * (seq2-seq2_mean)
    denominator = K.sqrt( K.mean(K.square(seq1-seq1_mean), axis=-1, keepdims=True) * K.mean(K.square(seq2-seq2_mean), axis=-1, keepdims=True) )
    PCC         = nominator / (denominator + K.common.epsilon())  # Pearson's correlation coefficient
    seq1_var    = K.mean(K.square(seq1-seq1_mean), axis=-1, keepdims=True)
    seq2_var    = K.mean(K.square(seq2-seq2_mean), axis=-1, keepdims=True)
    CCC         = K.constant(2.) * PCC * K.sqrt(seq1_var) * K.sqrt(seq2_var) / (seq1_var + seq2_var + K.square(seq1_mean - seq2_mean) + K.common.epsilon())
    CCC_loss    = K.constant(1.) - CCC
    return CCC_loss


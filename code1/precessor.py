import re
import subprocess

import joblib
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

def transform_ydata(df_column):
    y_cast = {True: [1, 0], False: [0, 1]}
    y_dataset = []
    for line in df_column:
        y_dataset.append(y_cast[line])
    return y_dataset
def seq2vector(seqs):
    seq_tensor = []
    for line in seqs:
        seq_tensor.append(line)

    return seq_tensor
def mfe_to_vector(mfe_value, seqs):
    mfe_tensor = []
    for line, seq in zip(mfe_value, seqs):
        mfe_tensor.append(line / len(seq))

    return mfe_tensor

def encode(seqs, strs):
    if not isinstance(seqs, list):
        print("[ERROR:encode] Input type must be multidimensional list.")
        return

    if len(seqs) != len(strs):
        print("[ERROR:encode] # sequences must be equal to # structures.")
        return

    encs = []
    for a_seq, a_str in zip(seqs, strs):
        encs.append([4 * i_seq + i_str + 1 for i_seq, i_str in zip(seq2num(a_seq), str2num(convloops(a_str)))])

    return encs

def one_hot_wrap(X_encs, MAX_LEN, DIM_ENC):
    num_X_encs = len(X_encs)
    X_encs_padded = pads(X_encs, maxlen=MAX_LEN, dtype='int8')
    X_encs_ = np.zeros(num_X_encs).tolist()
    for i in range(num_X_encs):
        X_encs_[i] = one_hot(X_encs_padded[i], DIM_ENC)

    return np.int32(X_encs_)

def import_cv_seqences(file_path):
    # dataframe = read_new_csv(file_path)
    dataframe = pd.read_csv(file_path)
    x_dataset = seq2vector(dataframe["mature_seq"])
    y_dataset = transform_ydata(dataframe["label"])
    mfes = mfe_to_vector(dataframe["MFE"], dataframe["mature_seq"])
    second_structs = dataframe['RNAFolds']
    # mfe = mfe_to_vector(dataframe["MFE"])
    # # transform into numpy array
    # x_dataset = np.array(x_dataset)
    y_dataset = np.array(y_dataset)
    # mfe_dataset = np.array(mfe)
    # print(x_dataset)
    # print(y_dataset)
    print("data vectorization finished!")
    return x_dataset, mfes, second_structs, y_dataset

# def import_pos_data(filename):
#     encs = []
#     seqs = import_seq(filename)
#     Y = to_categorical([0] * len(seqs), num_classes=2)
#     y_binary = np.argmax(Y, axis=1)
#     y = 1 - y_binary
#     strs,mfes = seq2str(seqs)
#     features = genFeature(seqs,y)
#     mfe_dataset = np.array(mfes)
#
#     mfe_2d = mfe_dataset[:, np.newaxis]
#
#
#     feature_dataset = np.concatenate((mfe_2d, features), axis=1)
#
#     return encode(seqs, strs),feature_dataset,Y

# def import_neg_data(filename):
#     encs = []
#     seqs = import_seq(filename)
#     Y = to_categorical([1] * len(seqs), num_classes=2)
#     y_binary = np.argmax(Y, axis=1)
#     y = 1 - y_binary
#     strs,mfes = seq2str(seqs)
#     features = genFeature(seqs,y)
#     mfe_dataset = np.array(mfes)
#
#     mfe_2d = mfe_dataset[:, np.newaxis]
#
#
#     feature_dataset = np.concatenate((mfe_2d, features), axis=1)
#     return encode(seqs, strs),feature_dataset,Y

def import_cv_data(file, MAX_LEN, DIM_ENC,signal):
    seqs,mfes, strs, Y_train = import_cv_seqences(file)

    y_binary = np.argmax(Y_train, axis=1)
    y = 1 - y_binary
    # strs, mfes = seq2str(seqs)
    mfe_dataset = np.array(mfes)
    mfe_2d = mfe_dataset[:, np.newaxis]

    feature_dataset = np.concatenate((mfe_2d, features), axis=1)
    X_dataset = encode(seqs, strs)
    X_train = one_hot_wrap(X_dataset, MAX_LEN, DIM_ENC)
    return X_train, Y_train, feature_dataset, bn_size


def fold5CV(train_file, val_file, MAX_LEN, DIM_ENC,signal):
    x_train_segment, y_train_segment, feature_train_segment, bn_size = \
        import_cv_data(train_file, MAX_LEN, DIM_ENC,signal)

    x_validation_segment, y_validation_segment, feature_validation_segment = \
        import_cv_val_data(val_file, MAX_LEN,DIM_ENC,signal)

    return x_train_segment, y_train_segment, x_validation_segment, \
           y_validation_segment, feature_train_segment, feature_validation_segment, bn_size
    # 20231228_sam++ <<
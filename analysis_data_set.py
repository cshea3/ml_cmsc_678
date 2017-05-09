import json
import argparse
import glob
import os
import shutil
import itertools as it
import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets, svm
import _pickle as cPickle

import queue 
from threading import Thread
from multiprocessing import Process, Queue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_data_pickle_files", help="path to normalized dataset")
    parser.add_argument("path_to_label_pickle_files", help="path to normalized dataset")
    parser.add_argument("filename", help="prefix file name")
    #parser.add_argument("path_to_all_possible_feature_values_filename", help="path to all possible feature values .json file ")
    #parser.add_argument("path_to_list_of_features_to_train", help="path to .txt file with list of features to use for training")
    #parser.add_argument("filename", help="output file name")
    #parser.add_argument("number_of_threads", help="number of threads to use")
    args = parser.parse_args()
    data=np.array([])
    labels=np.array([])
    
    #print("There are " +str(data) + " data samples")
    #print("There are " +str(labels) + " labels samples")
    print("load data from pickle files")
    for filename in glob.iglob(args.path_to_data_pickle_files + 'data_matrix'+'*.pckl', recursive=True):
         print(filename)
         #data=np.vstack(cPickle.load(open(filename,'rb')))
         with open(filename,'rb') as f:
             if len(data)==0:
                 data = cPickle.load(f)
             else:
                 #temp = cPickle.load(f)
                 #print(len(temp))
                 data = np.vstack((data,cPickle.load(f)))
             #print(data)
    print(data)
    print("load labels from pickle files")
    for filename in glob.iglob(args.path_to_label_pickle_files +'label_list'+'*.pckl', recursive=True):
        #label=np.vstack(cPickle.load(open(filename,'rb')))
        print(filename)
        with open(filename,'rb') as f:
            if len(labels) == 0:
                labels = cPickle.load(f)
            else:
                #temp=cPickle.load(f)
                #print(len(temp))
                labels=np.hstack((labels,cPickle.load(f)))
    print(labels)


   
    n_sample=len(data)
    np.random.seed(0)
    order = np.random.permutation(n_sample)
    data = data[order]
    labels = labels[order].astype(np.float)
    print(labels)

    data_train = data[:.9 * n_sample]
    labels_train = labels[:.9 * n_sample]
    data_test = data[.9 * n_sample:]
    labels_test = labels[.9 * n_sample:]
    with open('dump_of_data_train_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(data_train,fid, protocol=4)
    with open('dump_of_label_train_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(labels_train,fid, protocol=4)
    with open('dump_of_data_test_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(data_test,fid, protocol=4)
    with open('dump_of_label_test_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(labels_test,fid, protocol=4)
    

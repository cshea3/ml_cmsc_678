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
from sklearn.cross_validation import cross_val_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name_of_model", help="path to model")
    parser.add_argument("postfix", help="string after model")
    parser.add_argument("path_to_data_test_file", help="path to normalized dataset")
    parser.add_argument("path_to_label_test_file", help="path to normalized dataset")

    #parser.add_argument("filename", help="prefix file name")
    #parser.add_argument("svm_type", help="linear, poly or rfb")
    args = parser.parse_args()
    data_test = np.array([])
    label_test = np.array([])

    c_ =[0.1,1,10,100]
    score_values = np.array([])
    
    with open(args.path_to_data_test_file,'rb') as fid:
        data_test = cPickle.load(fid)
    with open(args.path_to_label_test_file,'rb') as fid:
        label_test = cPickle.load(fid)
    
    for x in c_ :
        with open(args.name_of_model+str(x)+postfix,'rb') as fid:
            clf = cPickle.load(fid)
        cross_score = cross_val_score(clf, data_test,label_test,cv=10)
        if len(score_values) == 0:
            score_values = cross_score
        else:
            score_value = np.vstack((score_values,cross_score))
        print("The cross validation scores are "+str(cross_score))
    print("Complete cross validation matrix is " + str(score_values))





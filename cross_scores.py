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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name_of_model", help="path to model")
    #parser.add_argument("path_to_label", help="path to labels")
    parser.add_argument("path_to_data_test_file", help="path to normalized dataset")
    parser.add_argument("path_to_label_test_file", help="path to normalized dataset")

    #parser.add_argument("filename", help="prefix file name")
    #parser.add_argument("svm_type", help="linear, poly or rfb")
    args = parser.parse_args()
    data_test = np.array([])
    label_test = np.array([])

    with open(args.name_of_model,'rb') as fid:
        clf = cPickle.load(fid)
    
    with open(args.path_to_data_test_file,'rb') as fid:
        data_test = cPickle.load(fid)
    with open(args.path_to_label_test_file,'rb') as fid:
        label_test = cPickle.load(fid)
    
    cross_score = clf.score(data_test,label_test)
    print("The cross validation scores are "+str(cross_score))

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
    for filename in glob.iglob(args.path_to_data_pickle_files +'*.pckl', recursive=True):
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
    for filename in glob.iglob(args.path_to_label_pickle_files +'*.pckl', recursive=True):
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
    #shape_l = labels.shape
    #labels_01 = np.zeros(shape=labels.shape)
    #np.copyto(labels_01,labels)
    #for the first attempt only look for class 0 and not class 0
    #for x in np.nditer(labels_01):
    #    if x > 0: labels_01[int(x)]=1

    #print(labels_01)
    #data_train_01 = data[:.9 * n_sample]
    #labels_train_01 = labels[:.9 * n_sample]
    #data_test_01 = data[.9 * n_sample:]
    #labels_test_01 = labels[.9 * n_sample:]

    #begin to trainup the dataset
    # TODO - train SVM using X and Y from 'data'
    #clf = svm.SVC(kernel='linear', gamma=10)
    #send the data for training
    #clf.fit(data_train_01, labels_train_01)
    #retrive the predicited labesls
    #predict_labels_01 = clf.predict(data_test_01)
    #with open('dump_of_svm01_'+str(args.filename)+'.pkl','wb') as fid:
    #    cPickle.dump(clf,fid, protocol=4)
    
    #with open('predicted_01'+str(args.filename)+'.pckl','wb') as pckle_f:
    #    cPickle.dump(predict_labels_01,pckle_f, protocol=4)

    ################## Linear
    print("Entering Linear SVM .....")
    clf = svm.SVC(kernel='linear',class_weight='balanced',C=.1)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_c.1_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid)
    predict_labels = clf.predict(data_test)
    with open('predicted_c.1_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f, protocol=4)
    
    clf = svm.SVC(kernel='linear',class_weight='balanced',C=1)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_c1_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid, protocol=4)
    predict_labels = clf.predict(data_test)
    with open('predicted_c1_'+str(args.filename)+'.pckl','wb') as pckle_f:
        cPickle.dump(predict_labels,pckle_f, protocol=4)


    clf = svm.SVC(kernel='linear',class_weight='balanced',C=10)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_c10_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid, protocol=4)
    predict_labels = clf.predict(data_test)
    with open('predicted_c10_'+str(args.filename)+'.pckl','wb') as pckle_f:
        cPickle.dump(predict_labels,pckle_f, protocol=4)


    clf = svm.SVC(kernel='linear',class_weight='balanced',C=100)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_c100_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid, protocol=4)
    predict_labels = clf.predict(data_test)
    with open('predicted_c100_'+str(args.filename)+'.pckl','wb') as pckle_f:
        cPickle.dump(predict_labels,pckle_f, protocol=4)

    ################## RBF
    print("Entering RBF SVM ....")
    clf = svm.SVC(kernel='rbf',class_weight='balanced',gamma=0.7, C=.1)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_rbf_c.1_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid, protocol=4)
    predict_labels = clf.predict(data_test)
    with open('predicted_rbf_c.1_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f, protocol=4)
 
    clf = svm.SVC(kernel='rbf',class_weight='balanced',gamma=0.7, C=1)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_rbf_c1_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid, protocol=4)
    predict_labels = clf.predict(data_test)
    with open('predicted_rbf_c1_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f, protocol=4)
     
    clf = svm.SVC(kernel='rbf',class_weight='balanced',gamma=0.7, C=10)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_rbf_c10_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid, protocol=4)
    predict_labels = clf.predict(data_test)
    with open('predicted_rbf_c10_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f, protocol=4)

    clf = svm.SVC(kernel='rbf',class_weight='balanced',gamma=0.7, C=100)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_rbf_c100_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid, protocol=4)
    predict_labels = clf.predict(data_test)
    with open('predicted_rbf_c100_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f, protocol=4)
 
    #################### Poly
    print("Entering Poly SVM ...." )
    clf = svm.SVC(kernel='poly',class_weight='balanced',gamma=0.7, C=.1)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_poly_c.1_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid, protocol=4)
    predict_labels = clf.predict(data_test)
    with open('predicted_poly_c.1_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f, protocol=4)
 
    clf = svm.SVC(kernel='poly',class_weight='balanced',gamma=0.7, C=1)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_poly_c1_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid, protocol=4)
    predict_labels = clf.predict(data_test)
    with open('predicted_poly_c1_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f, protocol=4)
     
    clf = svm.SVC(kernel='poly',class_weight='balanced',gamma=0.7, C=10)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_poly_c10_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid, protocol=4)
    predict_labels = clf.predict(data_test)
    with open('predicted_poly_c10_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f, protocol=4)

    clf = svm.SVC(kernel='poly',class_weight='balanced',gamma=0.7, C=100)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_poly_c100_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid, protocol=4)
    predict_labels = clf.predict(data_test)
    with open('predicted_poly_c100_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f, protocol=4)

    #plt.title('inear')
    #plt.show()
    #print("Classification report for classifier %s:\n%s\n"
    #  % (clf, metrics.classification_report(labels_test_01, predicted)))
    #print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels_test_01, predicted))

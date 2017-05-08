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
def svm_processing(svm_type, data_train, labels_train, data_test,labels_test, filename):
    c=[.01,.1,1,10,100]
    if svm_type == "linear" :
        print("Choosen type is Linear")
        for c_ in c:
            print("Creating svm with a c value of " + str(c_))
            clf = svm.SVC(kernel='linear',class_weight='balanced',C=float(c_))
            print("Finished creation, now onto fitting")
            clf.fit(data_train,labels_train)
            print("Finished fitting now need to save")
            with open('dump_of_svm_linear'+str(c_)+'_'+str(args.filename)+'.pkl','wb') as fid:
                cPickle.dump(clf,fid,protocol=4)
            print("Predicting the labels with a c value of "+ str(c_))
            predict_labels = clf.predict(data_test)
            with open('predicted_labels_linear_'+str(c_)+'_'+str(args.filename)+'.pckl','wb') as fid: 
                cPickle.dump(predict_labels,fid, protocol=4)
        print("End of the forloop")
    elif svm_type == "rbf":
        print("Entering RBF SVM ....")
        for c_ in c:
            print("Creating svm with a c value of " + str(c_))
            clf = svm.SVC(kernel='rbf',class_weight='balanced',gamma=0.7, C=float(c_))
            print("Finished creation, now onto fitting")

            clf.fit(data_train,labels_train)
            print("Finished fitting now need to save")

            with open('dump_of_svm_rbf_'+str(c_)+'_'+str(args.filename)+'.pkl','wb') as fid:
                cPickle.dump(clf,fid, protocol=4)
            print("Predicting the labels with a c value of "+ str(c_))
            predict_labels = clf.predict(data_test)
            with open('predicted_rbf_'+str(c_)+'_'+str(args.filename)+'.pckl','wb') as fid: 
                cPickle.dump(predict_labels,fid, protocol=4)
        print("End of the forloop")
    elif svm_type == "poly":
        print("Entering Poly SVM ...." )
        for c_ in c:
            print("Creating svm with a c value of " + str(c_))
            clf = svm.SVC(kernel='poly',class_weight='balanced',gamma=0.7, C=float(c_))
            print("Finished creation, now onto fitting")

            clf.fit(data_train,labels_train)
            print("Finished creation, now onto fitting")

            with open('dump_of_svm_poly_'+str(c_)+'_'+str(args.filename)+'.pkl','wb') as fid:
                cPickle.dump(clf,fid, protocol=4)
            print("Predicting the labels with a c value of "+ str(c_))
            predict_labels = clf.predict(data_test)
            with open('predicted_poly_'+str(c_)+'_'+str(args.filename)+'.pckl','wb') as fid: 
                cPickle.dump(predict_labels,fid, protocol=4)
        print("End of the forloop")

def svm_processing(svm_type, data_train, labels_train, data_test,labels_test, filename, c_):
    print("The value of C passed in is" + str(c_))
    if svm_type == "linear" :
        print("Creating svm with a c value of " + str(c_))
        clf = svm.SVC(kernel='linear',class_weight='balanced',C=float(c_))
        clf.fit(data_train,labels_train)
        with open('dump_of_svm_linear'+str(c_)+'_'+str(args.filename)+'.pkl','wb') as fid:
           cPickle.dump(clf,fid,protocol=4)
        print("Predicting the labels with a c value of "+ str(c_))
        predict_labels = clf.predict(data_test)
        with open('predicted_labels_linear_'+str(c_)+'_'+str(args.filename)+'.pckl','wb') as fid: 
            cPickle.dump(predict_labels,fid, protocol=4)
        print("End of the forloop")
    elif svm_type == "rbf":
        print("Entering RBF SVM ....")
        print("Creating svm with a c value of " + str(c_))
        clf = svm.SVC(kernel='rbf',class_weight='balanced',gamma=0.7, C=float(c_))
        clf.fit(data_train,labels_train)
        with open('dump_of_svm_rbf_'+str(c_)+'_'+str(args.filename)+'.pkl','wb') as fid:
            cPickle.dump(clf,fid, protocol=4)
        print("Predicting the labels with a c value of "+ str(c_))
        predict_labels = clf.predict(data_test)
        with open('predicted_rbf_'+str(c_)+'_'+str(args.filename)+'.pckl','wb') as fid: 
            cPickle.dump(predict_labels,fid, protocol=4)
    elif svm_type == "poly":
        print("Entering Poly SVM ...." )
        print("Creating svm with a c value of " + str(c_))
        clf = svm.SVC(kernel='poly',class_weight='balanced',gamma=0.7, C=float(c_))
        clf.fit(data_train,labels_train)
        with open('dump_of_svm_poly_'+str(c_)+'_'+str(args.filename)+'.pkl','wb') as fid:
            cPickle.dump(clf,fid, protocol=4)
        print("Predicting the labels with a c value of "+ str(c_))
        predict_labels = clf.predict(data_test)
        with open('predicted_poly_'+str(c_)+'_'+str(args.filename)+'.pckl','wb') as fid: 
            cPickle.dump(predict_labels,fid, protocol=4)
        print("End of the forloop")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_data_training_file", help="path to normalized dataset")
    parser.add_argument("path_to_label_training_file", help="path to normalized dataset")
    parser.add_argument("path_to_data_test_file", help="path to normalized dataset")
    parser.add_argument("path_to_label_test_file", help="path to normalized dataset")

    parser.add_argument("filename", help="prefix file name")
    parser.add_argument("svm_type", help="linear, poly or rfb")
    args = parser.parse_args()

    data_train=np.array([])
    labels_train=np.array([])
    data_test = np.array([])
    labels_test = np.array([])
    print("reading from the data_training")
    with open(args.path_to_data_training_file,'rb') as f:
        data_train = cPickle.load(f)
    print("The length of the training data is" + str(len(data_train)))
    with open(args.path_to_label_training_file,'rb') as f:
        labels_train=cPickle.load(f)
    print("The length of the training labels are" + str(len(labels_train)))
    with open(args.path_to_data_test_file, 'rb') as f:
        data_test = cPickle.load(f)
    print(len(data_test))
    with open(args.path_to_label_test_file, 'rb') as f:
        labels_test=cPickle.load(f)
    print(len(labels_train))
    
    
    threads_list = list()
    type_of_svm = ['linear','rbf','poly']
    c_ = [.01,.1,1,10,100]
    
    #for x in type_of_svm:
    for y in c_:
        t = Process(target=svm_processing, args=('linear', data_train, labels_train,data_test,labels_test,args.filename,y))
        t.start()
        threads_list.append(t)
    
    for t in threads_list:
        t.join()

    for y in c_:
        t = Process(target=svm_processing, args=('rbf', data_train, labels_train,data_test,labels_test,args.filename,y))
        t.start()
        threads_list.append(t)
    
    for t in threads_list:
        t.join()

    for y in c_:
        t = Process(target=svm_processing, args=('poly', data_train, labels_train,data_test,labels_test,args.filename,y))
        t.start()
        threads_list.append(t)
    
    for t in threads_list:
        t.join()

    

        






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
features_and_vector_lengths = {}

def display_data_matrix(data):
    print("\n========DATA MATRIX======")
    print(*data, sep='\n')

    print("~~~~~~~~~~~~~")
    X = [row[:-1] for row in data]
    Y = [row[-1] for row in data]

    print("X: ")
    for x in X:
        print(x)
    print("Y: ")
    for y in Y:
        print(y)
    print("========DATA SUMMARY======")
    print("matrix dimensions: " + str(len(data)) + " x " + str(len(data[0])))
    print("number of rows: " + str(len(data)))
    print("length of feature vector: " + str(len(X[0])))

def find_trends_in_data(data,labels):
    #take the full data find the ones that have the same labels
    indices_0 = [i for i, x in enumerate(labels) if x == 0]
    indices_1 = [i for i, x in enumerate(labels) if x == 1]
    indices_2 = [i for i, x in enumerate(labels) if x == 2]
    indices_3 = [i for i, x in enumerate(labels) if x == 3]
    #sum the 

def convert_label(label_as_list):
    # index of where "1" indicates the class (0,1,2,3)
    class_label = [label_as_list.index(1)]
    return class_label 

def get_all_features_and_vector_lengths(path_to_all_possible_feature_values_filename):
    # open json file containing all possible features and feature_values
    print("ALL POSSIBLE FEATURES AND COUNT OF FEATURE-VALUES IN DATASET:")
    num_columns = 0
    num_rows = 0
    with open(path_to_all_possible_feature_values_filename) as data_file:
        data_set_features_values = json.load(data_file)
        for feature in data_set_features_values:
            print(feature + ": " + str(len(data_set_features_values[feature])))
            features_and_vector_lengths[feature] = len(data_set_features_values[feature])
            if num_columns < len(data_set_features_values[feature]):
                num_columns = len(data_set_features_values[feature])
    #num_rows = len(data_set_features_values)
    return num_columns,num_rows

def build_matrix_from_selected_features_list(path_to_dataset,list_of_dataset, path_to_list_of_features_to_train, max_columns, queue_data, queue_label, queue_feature):
    features_to_use = sorted([line.rstrip() for line in open(path_to_list_of_features_to_train)])
    feature_vector_count = np.array([])
    # 2D matrix that can be used in the model
    data_matrix = np.array([])
    label = np.array([])
    count = 0
    #print("IN function here is the work list" + str(list_of_dataset))
    for folder_name in list_of_dataset:
        for filename in glob.iglob(path_to_dataset + folder_name, recursive=True):
            with open(filename) as data_file:
                print("PROCESSING: " + filename)
                malware_sample = json.load(data_file)
                feature_vector_count = np.array([])
                matrix = np.array([])
                row = np.array([])
                for feature in features_to_use:
                    if count==0: feature_vector_count = np.hstack((feature_vector_count, np.array([features_and_vector_lengths[feature]]))) 
                    #before processing find the feature set with the
                    # if feature is not present in sample, pad feature vector with zeros
                    if feature not in malware_sample:
                        vector_len = features_and_vector_lengths[feature]
                        #print("  " + feature + ": " + "NOT present in " + filename + " (padding feature vector with " + str(vector_len) + " zeros)")
                        temp = np.zeros(vector_len)
                        row = np.hstack((row,temp))
                        #to build a 2D matrix use this
                        #if(matrix.size ==0):
                        #    matrix=temp
                        #else:
                        #    matrix=np.vstack((matrix,temp))
                    else:
                        features = np.array(malware_sample[feature])
                        #using a 2D matrix set max row of zeros
                        #temp = np.zeros(50)
                        #copy in the features starting at the 0 position 
                        #temp[:len(features)]=features
                        #if(matrix.size ==0):
                        #    matrix=temp
                        #else:
                        #    matrix=np.vstack((matrix,temp))
                        row=np.hstack((row,features))
                    #print(row)    
                class_label = convert_label(malware_sample['label'])
                label=np.hstack((label,class_label))

                # append X to matrix
                if data_matrix.size == 0:
                   data_matrix=row
                else:
                   data_matrix=np.vstack((data_matrix,row))
            if count == 0: print(feature_vector_count)
            count = 1 
    print("Placing data in queues") 

    print("Data placed in queues.")		

def build_matrix_from_selected_features_list_file(path_to_dataset,list_of_dataset, path_to_list_of_features_to_train, max_columns, thread_num, postfix):
    features_to_use = sorted([line.rstrip() for line in open(path_to_list_of_features_to_train)])
    feature_vector_count = np.array([])
    # 2D matrix that can be used in the model
    data_matrix = np.array([])
    label = np.array([])
    count = 0
    #print("IN function here is the work list" + str(list_of_dataset))
    for folder_name in list_of_dataset:
        for filename in glob.iglob(path_to_dataset + folder_name, recursive=True):
            with open(filename) as data_file:
                print("PROCESSING: " + filename)
                malware_sample = json.load(data_file)
                feature_vector_count = np.array([])
                matrix = np.array([])
                row = np.array([])
                for feature in features_to_use:
                    if count==0: feature_vector_count = np.hstack((feature_vector_count, np.array([features_and_vector_lengths[feature]]))) 
                    #before processing find the feature set with the
                    # if feature is not present in sample, pad feature vector with zeros
                    if feature not in malware_sample:
                        vector_len = features_and_vector_lengths[feature]
                        #print("  " + feature + ": " + "NOT present in " + filename + " (padding feature vector with " + str(vector_len) + " zeros)")
                        temp = np.zeros(vector_len)
                        row = np.hstack((row,temp))
                        #to build a 2D matrix use this
                        #if(matrix.size ==0):
                        #    matrix=temp
                        #else:
                        #    matrix=np.vstack((matrix,temp))
                    else:
                        features = np.array(malware_sample[feature])
                        #using a 2D matrix set max row of zeros
                        #temp = np.zeros(50)
                        #copy in the features starting at the 0 position 
                        #temp[:len(features)]=features
                        #if(matrix.size ==0):
                        #    matrix=temp
                        #else:
                        #    matrix=np.vstack((matrix,temp))
                        row=np.hstack((row,features))
                    #print(row)    
                class_label = convert_label(malware_sample['label'])
                label=np.hstack((label,class_label))

                # append X to matrix
                if data_matrix.size == 0:
                   data_matrix=row
                else:
                   data_matrix=np.vstack((data_matrix,row))
            if count == 0: print(feature_vector_count)
            count = 1 
    with open('data_matrix_'+str(thread_num)+'_'+postfix+'.pckl','wb') as file_h:
        cPickle.dump(data_matrix,file_h)
    with open('label_list_'+str(thread_num)+'_'+postfix+'.pckl','wb') as file_h:
        cPickle.dump(label,file_h)
        		 
def build_matrix_from_selected_features(path_to_dataset, path_to_list_of_features_to_train, max_columns):
    # read in desired features for use in the model from text file and sort alphabetically
    features_to_use = sorted([line.rstrip() for line in open(path_to_list_of_features_to_train)])

    print("-----------")
    print("FEATURES USED FOR THIS MODEL:", *features_to_use, sep='\n')
    print("-----------")
    feature_vector_count = np.array([])
    # 2D matrix that can be used in the model
    data_matrix = np.array([])
    label = np.array([])
    count = 0
    for filename in glob.iglob(path_to_dataset + '/**/*.json', recursive=True):
        with open(filename) as data_file:
            print("PROCESSING: " + filename)
            malware_sample = json.load(data_file)
            feature_vector_count = np.array([])
            matrix = np.array([])
            row = np.array([])
            for feature in features_to_use:
                if count==0: feature_vector_count = np.hstack((feature_vector_count, np.array([features_and_vector_lengths[feature]]))) 
                #before processing find the feature set with the
                # if feature is not present in sample, pad feature vector with zeros
                if feature not in malware_sample:
                    vector_len = features_and_vector_lengths[feature]
                    #print("  " + feature + ": " + "NOT present in " + filename + " (padding feature vector with " + str(vector_len) + " zeros)")
                    temp = np.zeros(vector_len)
                    row = np.hstack((row,temp))
                    #to build a 2D matrix use this
                    #if(matrix.size ==0):
                    #    matrix=temp
                    #else:
                    #    matrix=np.vstack((matrix,temp))
                    
                else:
                    features = np.array(malware_sample[feature])
                    #using a 2D matrix set max row of zeros
                    #temp = np.zeros(50)
                    #copy in the features starting at the 0 position 
                    #temp[:len(features)]=features
                    #if(matrix.size ==0):
                    #    matrix=temp
                    #else:
                    #    matrix=np.vstack((matrix,temp))
                    row=np.hstack((row,features))
            class_label = convert_label(malware_sample['label'])
            label=np.hstack((label,class_label))

            # append X to matrix
            if data_matrix.size == 0:
                data_matrix=row
            else:
                data_matrix=np.vstack((data_matrix,row))
        if count == 0: print(feature_vector_count)
        count = 1 #no need to recalc the number of instances in each part of the feature vector        
    #print(data_matrix)
    #print("Final Matrix is " + str(data_matrix))
    return data_matrix,label,feature_vector_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_normalized_dataset", help="path to normalized dataset")
    parser.add_argument("path_to_all_possible_feature_values_filename", help="path to all possible feature values .json file ")
    parser.add_argument("path_to_list_of_features_to_train", help="path to .txt file with list of features to use for training")
    parser.add_argument("filename", help="output file name")
    parser.add_argument("number_of_threads", help="number of threads to use")
    args = parser.parse_args()
    data=np.array([])
    labels=np.array([])
    max_columns = get_all_features_and_vector_lengths(args.path_to_all_possible_feature_values_filename)
    #data,labels,feature_vector_count = build_matrix_from_selected_features(args.path_to_normalized_dataset, args.path_to_list_of_features_to_train, max_columns)

    with open('dump_of_data_vector_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(data,fid)

    with open('dump_of_labels_vector_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(labels,fid)

    onlyFolders = [f for f in os.listdir(args.path_to_normalized_dataset)]
    number_of_folders = int(len(onlyFolders))
    folders_per_thread = number_of_folders/(int(args.number_of_threads))
    #print(" number of folders per thread is "+str(folders_per_thread))
    #remainder = 0
    threads = int(args.number_of_threads)
    if (number_of_folders%(threads) != 0):
        remainder = number_of_folders%threads
    
    #[[] for dummy in xrange(int(args.number_of_threads))]
    work_list = [onlyFolders[i:i+int(folders_per_thread)] for i in range(0, len(onlyFolders), int(folders_per_thread))]
    print(" HERE IS THE WORK LIST LENGTH" + str(len(work_list)))
    #speed up the dict creation
    #list_of_data_queues = [] 
    #list_of_label_queues = []
    #for x in range(0,int(args.number_of_threads)):
    #    list_of_data_queues.append(Queue())
    #    list_of_label_queues.append(Queue())

    #que_feature_vector_count = Queue()
   
    threads_list = list()
    #build_matrix_from_selected_features_list_file(path_to_dataset,list_of_dataset, path_to_list_of_features_to_train, max_columns, thread_num, postfix):
    for idx, x in enumerate(work_list):
        t = Process(target=build_matrix_from_selected_features_list_file, args=(args.path_to_normalized_dataset, x, args.path_to_list_of_features_to_train, 0,idx,args.filename))
        t.start()
        threads_list.append(t)
   
    #for x in range(0,int(args.number_of_threads)):
    #    #while not list_of_data_queues[x].empty():
    #   print("waiting for data....")
    #    data=np.append(data,list_of_data_queues[x].get(True))
    #    print(data)
    #print("Received data is " + str(data))

    #for x in range(0,int(args.number_of_threads)):
    #    #while not list_of_label_queues[x].empty():
    #    print("waiting for labels....")
    #    label=np.append(label,list_of_label_queues[x].get(True))

    #print("Received labels are " + str(labels))
    for t in threads_list:
        t.join()
 

import json
import argparse
import glob
import os
import shutil
import itertools as it
import numpy as np
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
		 
def build_matrix_from_selected_features(path_to_dataset, path_to_list_of_features_to_train, max_columns):
    # read in desired features for use in the model from text file and sort alphabetically
    features_to_use = sorted([line.rstrip() for line in open(path_to_list_of_features_to_train)])

    print("-----------")
    print("FEATURES USED FOR THIS MODEL:", *features_to_use, sep='\n')
    print("-----------")

    # 2D matrix that can be used in the model
    data_matrix = np.array([])
    label = np.array([])
    for filename in glob.iglob(path_to_dataset + '/**/*.json', recursive=True):
        with open(filename) as data_file:
            print("PROCESSING: " + filename)
            malware_sample = json.load(data_file)

            matrix = np.array([])
            #matrix=np.array([])
            row = np.array([])
            for feature in features_to_use:
		#before processing find the feature set with the
                # if feature is not present in sample, pad feature vector with zeros
                if feature not in malware_sample:
                    vector_len = features_and_vector_lengths[feature]
                    print("  " + feature + ": " + "NOT present in " + filename + " (padding feature vector with " + str(vector_len) + " zeros)")
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
            #print(data_matrix)
    print("Final Matrix is " + str(data_matrix))
    return data_matrix,label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_normalized_dataset", help="path to normalized dataset")
    parser.add_argument("path_to_all_possible_feature_values_filename", help="path to all possible feature values .json file ")
    parser.add_argument("path_to_list_of_features_to_train", help="path to .txt file with list of features to use for training")
    args = parser.parse_args()
    data=np.array([])
    labels=np.array([])
    max_columns = get_all_features_and_vector_lengths(args.path_to_all_possible_feature_values_filename)
    data,labels = build_matrix_from_selected_features(args.path_to_normalized_dataset, args.path_to_list_of_features_to_train, max_columns)

    # print out data (for debugging)
    display_data_matrix(data)
    print(labels)
    #X = [row[:-1] for row in data]
    #Y = [row[-1] for row in data]

    # TODO - train SVM using X and Y from 'data'

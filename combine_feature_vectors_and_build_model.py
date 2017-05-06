import json
import argparse
import glob
import os
import shutil
import logging
import itertools as it
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

features_and_vector_lengths = {}


def display_accuracy(scores):
    print("cross validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def display_data_matrix(data):
    logging.debug("\n========DATA MATRIX======")
    #logging.debug(*data, sep='\n')
    for row in data:
        logging.debug(row)

    logging.debug("~~~~~~~~~~~~~")
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]

    logging.debug("X: ")
    for x in X:
        logging.debug(x)
    logging.debug("y: ")
    logging.debug(y)

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
    logging.info("ALL POSSIBLE FEATURES AND COUNT OF FEATURE-VALUES IN DATASET:")
    logging.info("-----------")
    with open(path_to_all_possible_feature_values_filename) as data_file:
        data_set_features_values = json.load(data_file)
        for feature in data_set_features_values:
            logging.info(feature + ": " + str(len(data_set_features_values[feature])))
            features_and_vector_lengths[feature] = len(data_set_features_values[feature])
       
def build_matrix_from_selected_features(path_to_dataset, path_to_list_of_features_to_train):
    # read in desired features for use in the model from text file and sort alphabetically
    features_to_use = sorted([line.rstrip() for line in open(path_to_list_of_features_to_train)])

    logging.info("\n")
    #logging.debug("FEATURES USED FOR THIS MODEL:", *features_to_use, sep='\n')
    logging.info("FEATURES USED FOR THIS MODEL:")
    logging.info("-----------")
    for f in features_to_use:
        logging.info(f)
    logging.info("-----------")


    # 2D matrix that can be used in the model
    data_matrix = []

    for filename in glob.iglob(path_to_dataset + '/**/*.json', recursive=True):
        with open(filename) as data_file:
            logging.debug("PROCESSING: " + filename)
            malware_sample = json.load(data_file)

            row = []
            for feature in features_to_use:

                # if feature is not present in sample, pad feature vector with zeros
                if feature not in malware_sample:
                    vector_len = features_and_vector_lengths[feature]
                    logging.debug("  " + feature + ": " + "NOT present in " + filename + " (padding feature vector with " + str(vector_len) + " zeros)")
                    temp = list(map(lambda x: 0, range(vector_len)))     
                    row.extend(temp)
                    #logging.debug("  row: " + str(row))
                else:
                    logging.debug("  " + feature + ":" + str(malware_sample[feature]))
                    # collapse all features into one list (X)
                    row.extend(malware_sample[feature])
                    #logging.debug("row: " + str(row))

            class_label = convert_label(malware_sample['label'])
            row.extend(class_label)
            logging.debug("  label: " + str(class_label))
            logging.debug("  row: " + str(row))

            # append X to matrix
            data_matrix.append(row)

    return data_matrix


def train_svm(X, y):
    print("\n*******************************")
    print("*******************************")
    print("Starting to train with SVM...")
    print("*******************************")
    print("*******************************")
    
    # convert to numpy ndarray (probably not even necessary)
    X = np.array(X)
    y = np.array(y)
    
    # print original dataset feature values
    print("dataset (X):")
    print(X)
    print('{}: {}'.format("length of feature vector", len(X[0])))
    print()
    print("classification (y):")
    print(y)
    print('{}: {}'.format("number of classes", len(set(y))))
    print('{}: {}'.format("number of samples", len(y)))
    
    print("*************\n")
    
    # account for 'unbalanced' with class_weight
    clf = svm.SVC(kernel='linear', class_weight='balanced', C = 1.0)

    # cross validation scores (10-fold)
    scores = cross_val_score(clf, X, y, cv=10)
    print("SCORES:")
    print(scores)
    display_accuracy(scores)
    print()

    clf.fit(X,y)

    print('{} {}'.format("clf:", clf))
    print()
    
    w = clf.coef_[0]
    print("clf.coef_[0] (w):")
    print("feature vector length: " + str(len(clf.coef_[0])))
    print(w)
    print()
    
    print('{}: {}'.format("clf.intercept_[0]", clf.intercept_[0]))
    print('{}: {}'.format("number of classes", len(clf.n_support_)))
    print('{}: {}'.format("number of support vectors for each class:", clf.n_support_))
    
    print("support vectors:")
    print(clf.support_vectors_)
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_normalized_dataset", help="path to normalized dataset")
    parser.add_argument("path_to_all_possible_feature_values_filename", help="path to all possible feature values .json file ")
    parser.add_argument("path_to_list_of_features_to_train", help="path to .txt file with list of features to use for training")
    parser.add_argument("--log", help="log level")
    args = parser.parse_args()

    if args.log:
        numeric_level = getattr(logging, args.log.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)
        logging.basicConfig(filename='log.txt', filemode='w', level=numeric_level, format='%(levelname)s: %(message)s')

    get_all_features_and_vector_lengths(args.path_to_all_possible_feature_values_filename)
    data = build_matrix_from_selected_features(args.path_to_normalized_dataset, args.path_to_list_of_features_to_train)

    # print out data (for debugging)
    display_data_matrix(data)

    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]

    # train SVM using X and Y from 'data'
    train_svm(X, y)
    

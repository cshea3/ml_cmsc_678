import json
import argparse
import glob
import os
import shutil
import itertools as it
import numpy as np
from multiprocessing import Process
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

def find_features(path_to_dataset,label_name,path_to_list_of_all_features,write_to_file):
    #extract the key values from the dictionary
    with open(path_to_list_of_all_features) as file_:
        features_to_use = json.load(file_)
        features_to_use=features_to_use.keys()
    #print(features_to_use)
    count = 0
    feature_list = np.array([])
    matrix = np.array([])            
    #open all the json files in the directory
    for filename in glob.iglob(path_to_dataset + '/**/*.json', recursive=True):
        with open(filename) as data_file:
            #load the 
            malware_sample = json.load(data_file)
            #find the label we are looking for
            value=label_name == np.array(malware_sample['label'])
            #print(value)               
            if (value.all()):
                print("PROCESSING: " + filename)
                row=np.array([])
                for feature in features_to_use:
                    if count == 0:
                        feature_list=np.append(feature_list,feature)
                    if feature not in malware_sample:
                        #row = np.append(row,label)
                        row = np.append(row,0)
                    else:
                        row = np.append(row,np.sum(np.array(malware_sample[feature])))
                #matrix = np.vstack((matrix,row))
                if matrix.size == 0: matrix=row
                else: matrix=np.vstack((matrix,row))
        count = 1
    #sum up all the columns in the matrix
    sumation=np.sum(matrix,axis=0)
    
    apt_file = open(write_to_file,'w')
    json.dump(matrix.tolist(),write_to_file)
    json.dump(feature_list.tolist(),write_to_file)
    apt_file.close()

    return sumation,feature_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_normalized_dataset", help="path to normalized dataset")
    parser.add_argument("path_to_all_possible_feature_values_filename", help="path to all possible feature values .json file ")
    #parser.add_argument("label", help="name of classification")
    args = parser.parse_args()                
    




    
    class_type=np.array([])
    type_zeus = np.zeros((4),dtype=float)
    type_zeus[0]=1
    class_type=type_zeus

    type_locker = np.zeros((4),dtype=float)
    type_locker[1]=1
    class_type=np.vstack((class_type,type_locker))
    
    type_crypto = np.zeros((4),dtype=float)
    type_crypto[2]=1
    class_type=np.vstack((class_type,type_crypto))
    
    type_apt1 = np.zeros((4),dtype=float)
    type_apt1[3]=1
    class_type=np.vstack((class_type,type_apt1))
    p=np.array([])

    #create list of files to write to
    file_list = np.array([])
    for iter_ in range(0,4):
        v = 'feature_file_'+str(iter_)+'.json'
        #file_list = np.append(file_list,
        print(v)
        print(class_type)
    for iter_ in range(0,4):     
        p=np.append(p,Process(target=find_features, args=(args.path_to_normalized_dataset,class_type[iter_],args.path_to_all_possible_feature_values_filename,v[iter_])))
        print(p)
	
        p[int(iter_)].start()
        #print(index)
        #print(elem)
    for iter_ in range(0,4):
        p[iter_].join()
        print(iter_)

    
    #values,names=find_features(args.path_to_normalized_dataset,type_apt1,args.path_to_all_possible_feature_values_filename)
    
    #y_pos = np.arange(len(names))

    #plt.bar(y_pos,values, align='center', alpha=0.05)
    #plt.xticks(y_pos, names)
    #plt.show()

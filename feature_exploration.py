import json
import argparse
import glob
import os
import shutil
import itertools as it
import numpy as np
from multiprocessing import Process
import operator

def find_features(path_to_dataset,label_name,path_to_list_of_all_features,write_to_file):
    #label_name = np.flip(label_name,0)
    #extract the key values from the dictionary
    with open(path_to_list_of_all_features) as file_:
        features_to_use = json.load(file_)
        features_to_use=features_to_use.keys()
    #print(features_to_use)
    count = 0
    feature_list = np.array([])
    matrix = np.array([],dtype='d')           
    #open all the json files in the directory
    for filename in glob.iglob(path_to_dataset + '/**/*.json', recursive=True):
        with open(filename) as data_file:
            #load the 
            malware_sample = json.load(data_file)
            #find the label we are looking for
            value=label_name == np.array(malware_sample['label'])
            #print(" the label name is " + str(label_name))
            #print(" the recovered label is " + str(np.array(malware_sample['label'])))
            #print(value)
            row=np.array([])               
            if (value.all()):
                print("PROCESSING: " + filename)
                #print("The count is !!!!! " + str(count))
                for feature in features_to_use:
                    if count == 0:
                        feature_list=np.append(feature_list,feature)
                        #print(feature_list)
                    if feature not in malware_sample:
                        row = np.append(row,0)
                    else:
                        row = np.append(row,np.sum(np.array(malware_sample[feature])))
                count = 1
        if row.size != 0:
            if matrix.size == 0: matrix=row
            else: matrix=np.vstack((matrix,row))
   
    #sum up all the columns in the matrix
    print(" The matrix is " + str(matrix))
    sumation=np.array([],dtype='d')
    sumation=np.sum(matrix,axis=0,dtype='d')
    print(feature_list.size)
    print("the summation is " + str(sumation.tolist()))
    #create dictionary
    #print(type(feature_list.tolist()))
    #print(type(sumation.tolist()))
    #dictionary = dict(list(zip(feature_list.tolist(), sumation.tolist())))
    dictionary = {}
    for i in range(0,sumation.size):
        dictionary[feature_list[i]] = sumation[i]
        
    print(type(write_to_file))
    print("what is the type of dictionary" + str(dictionary))
    print("what is write_to_file " + str(dictionary))
    value_file = open(write_to_file,'w')
    json.dump(dictionary,value_file)
    #json.dump(feature_list.tolist(),value_file)
    value_file.close()

    #return dictionary

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
    #p=np.array([])

    find_features(args.path_to_normalized_dataset,[0,0,0,1],args.path_to_all_possible_feature_values_filename,'features_file_apt1.json')

    find_features(args.path_to_normalized_dataset,[0,0,1,0],args.path_to_all_possible_feature_values_filename,'features_file_crypto.json')

    find_features(args.path_to_normalized_dataset,[0,1,0,0],args.path_to_all_possible_feature_values_filename,'features_file_locker.json')

    find_features(args.path_to_normalized_dataset,[1,0,0,0],args.path_to_all_possible_feature_values_filename,'features_file_zeus.json')


    #create list of files to write to
    #file_list = np.array([])
    #for iter_ in range(0,4):
    #    file_list=np.append(file_list,'feature_file_'+str(iter_)+'.json')
    #file_list = np.append(file_list,
    #print(file_list)
    

    #for iter_ in range(0,4):     
    #    p=np.append(p,Process(target=find_features, args=(args.path_to_normalized_dataset,class_type[iter_],args.path_to_all_possible_feature_values_filename,file_list[iter_])))

    #    print(p)
    #    p[int(iter_)].start()
    #for iter_ in range(0,4):
    #    p[iter_].join()
    #    print(iter_)

    
    #values,names=find_features(args.path_to_normalized_dataset,type_apt1,args.path_to_all_possible_feature_values_filename)
    
    #y_pos = np.arange(len(names))

    #plt.bar(y_pos,values, align='center', alpha=0.05)
    #plt.xticks(y_pos, names)
    #plt.show()


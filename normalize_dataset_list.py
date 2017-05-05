import os
import shutil
from os import listdir
from os.path import isfile, join
from sys import executable
import subprocess
from subprocess import Popen, PIPE
import argparse
from multiprocessing import Process
import json 
import numpy as np
import glob
#https://www.ibm.com/developerworks/aix/library/au-multiprocessing/index.html

def parse_files(path_set,destination_dir,list_of_files,data_set_features_values):
    print("IN the parse_file")
    for items in list_of_files:
        print(path_set+items)
        for filename in glob.iglob(path_set+items+'/*.json',recursive=True):
            print(filename)
            with open(filename) as data_file:
                data_to_normalize = json.load(data_file)
                
                # include only Zeus, Crypto, Locker, and APT1 
                # and ignore other malware families
                lbl = data_to_normalize["properties"]["label"]
                if not lbl == "APT1" and \
                  not lbl == "Crypto" and \
                  not lbl == "Locker" and \
                  not lbl == "Zeus":
                    print("SKIPPING: " + filename)
                    continue

            print("NORMALIZING: " + filename)
            
            temp_normalized_dict = {}
            properties = sorted((data_to_normalize["properties"]).keys())
            for feature in properties:
                # get vector length of this feature
                vector_len = len(data_set_features_values[feature])

                # initialize to all zeros  
                temp_normalized_dict[feature] = list(map(lambda x: 0, range(vector_len)))
    
                # set indices to 0 or 1
                feature_values_str = data_to_normalize["properties"][feature]
                feature_values_list = [x for x in feature_values_str.split()]
                for f_value in feature_values_list:
                    # get index from dataset
                    idx = data_set_features_values[feature].index(f_value)

                    #print("feature: " + feature + " : " + "value " + f_value + " idx: " + str(idx))

                    # set index to "1" since this feature-value is present
                    temp_normalized_dict[feature][idx] = 1
            
            # output normalized sample to file in new directory
            normalized_filename =  os.path.splitext(os.path.basename(filename))[0] + "__normalized" + ".json"
            normalized_filepath = os.path.join(destination_dir, normalized_filename)
            with open(normalized_filepath, 'w') as outfile:
                outfile.write(json.dumps(temp_normalized_dict, sort_keys=True, indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_dataset", help="path to dataset")
    parser.add_argument("all_possible_feature_value_filename", help="path to feature-values")
    parser.add_argument("destination_dir", help="output directory")
    parser.add_argument("number_of_threads", help="number of worker threads")
    args = parser.parse_args()
    #malwareFolders = [f for f in listdir("/data/cwshea/ml_cmsc_678/MalwareTrainingSets/trainingSets")]
    #input('Enter to exit from this launcher script...')

    # create output directory of normalized dataset
    # delete directory first if already exists
    print("CREATING OUTPUT DIRECTORY FOR NORMALIZED DATASET: " +  args.destination_dir)
    if os.path.exists(args.destination_dir):
        print("\tdirectory already exists - overwriting...")
        shutil.rmtree(args.destination_dir)
    os.mkdir(args.destination_dir)
    
    with open(args.all_possible_feature_value_filename) as data_file:
        data_set_features_values = json.load(data_file)
        for feature in data_set_features_values:
            print(feature + ": " + str(len(data_set_features_values[feature])))
    print("--------------------------------------------------")

    onlyFolders = [f for f in listdir(args.path_to_dataset)]
    number_of_folders = int(len(onlyFolders))
    folders_per_thread = number_of_folders/(int(args.number_of_threads))
    print(" number of folders per thread is "+str(folders_per_thread))
    remainder = 0
    threads = int(args.number_of_threads)
    if (number_of_folders%(threads) != 0):
        remainder = number_of_folders%threads
    
    #[[] for dummy in xrange(int(args.number_of_threads))]
    work_list = [onlyFolders[i:i+int(folders_per_thread)] for i in range(0, len(onlyFolders), int(folders_per_thread))]
    p=np.array([])
    print(len(work_list))
    print(len(p))
    print(type(p))
    for index,elem in enumerate(work_list):     
        p=np.append(p,Process(target=parse_files, args=(args.path_to_dataset,args.destination_dir,elem,data_set_features_values)))
        print(p)
        p[int(index)].start()
        #print(index)
        #print(elem)
    for index2,elem2 in enumerate(work_list):
        p[index2].join()
        print(index2)
    #print work_list
    #print(len(work_list))
    #cpid = os.fork()
    #if not cpid:
    #    import somescript
    #    os._exit(0)
        
    #os.waitpid(cpid, 0)
    #print("call before the Popen")
    #process=subprocess.Popen(['python /data/cwshea/ml_cmsc_678/normalize_dataset_input.py ']+item, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    #stdout,stderr = process.communicate()
    #return_code = process.poll()
    #print("stdout='{}'\nstderr='{}'\nreturn_code='{}'".format(stdout, stderr, return_code))
    #print("call after Popen")
    #p.communicate()
    #parse_files(path_set,list_of_files)

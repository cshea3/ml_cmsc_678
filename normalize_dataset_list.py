import os
import shutil
from os import listdir
from os.path import isfile, join
from sys import executable
import subprocess
from subprocess import Popen, PIPE
import argparse

#https://www.ibm.com/developerworks/aix/library/au-multiprocessing/index.html
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
    
    onlyFolders = [f for f in listdir(args.path_to_dataset)]
    number_of_folders = int(len(onlyFolders))
    folders_per_thread = number_of_folders/(int(args.number_of_threads))
    print(" number of folders per thread is "+str(folders_per_thread))
    remainder = 0
    threads = int(args.number_of_threads)
    if (number_of_folders%(threads) != 0):
        remainder = number_of_folders%threads
    
    #[[] for dummy in xrange(int(args.number_of_threads))]
    work_list = [onlyFolders[i:i+folders_per_thread] for i in range(0, len(onlyFolders), folders_per_thread)]
    for item in work_list:
        #cpid = os.fork()
        #if not cpid:
        #    import somescript
        #    os._exit(0)
        
    #os.waitpid(cpid, 0)
        print("call before the Popen")
        process=subprocess.Popen(['python /data/cwshea/ml_cmsc_678/normalize_dataset_input.py ']+item, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
	stdout,stderr = process.communicate()
        return_code = process.poll()
        print("stdout='{}'\nstderr='{}'\nreturn_code='{}'".format(stdout, stderr, return_code))
        print("call after Popen")
        #p.communicate()
    
    #print work_list
    #print(len(work_list))
    p_status = p.wait()
    

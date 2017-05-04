import json
import argparse
import glob
import os
import shutil

data_set_features_values = {}
destination_dir = "dataset_normalized"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_dataset", help="path to dataset")
    parser.add_argument("all_possible_feature_value_filename", help="path to feature-values")
    args = parser.parse_args()

    # open json file containing all possible features and feature_values
    print("All possible features and count of feature-values in dataset:")
    with open(args.all_possible_feature_value_filename) as data_file:
        data_set_features_values = json.load(data_file)

        for feature in data_set_features_values:
            print(feature + ": " + str(len(data_set_features_values[feature])))
    print("--------------------------------------------------")

    # create output directory of normalized dataset
    # delete directory first if already exists
    print("CREATING OUTPUT DIRECTORY FOR NORMALIZED DATASET: " +  destination_dir)
    if os.path.exists(destination_dir):
        print("\tdirectory already exists - overwriting...")
        shutil.rmtree(destination_dir)
    os.mkdir(destination_dir)

    # open every .json file and write normalized version to new folder
    for filename in glob.iglob(args.path_to_dataset + '/**/*.json', recursive=True):
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

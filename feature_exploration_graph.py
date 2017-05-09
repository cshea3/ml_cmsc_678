import json 
import argparse

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import operator
def plot_analysis():
    print("plot_analysis")

def analyize_files(dictionary):
    print("plot_analysis")
    objects = dictionary.keys()
    y_pos = np.arange(len(objects))
    performance = dictionary.values()
 
    print(objects)
    print(performance)


    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Occurance')
    plt.title('Feature Usage Analyis')
 
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("path_to_dataset", help="path to dataset")
    parser.add_argument("file_name", help="file name")
    parser.add_argument("top_occurances", help="what number of iterms should be reported")

    args = parser.parse_args()
    
    with open(args.file_name) as data_file:
        data_to_analysize = json.load(data_file)
        objects = data_to_analysize.keys()
        y_pos = np.arange(len(objects))
        performance = data_to_analysize.values()
 
        print(objects)
        print(performance)


        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Occurance')
        plt.title('Feature usage')
 
        plt.show()    
        sorted_x = [(k,data_to_analysize[k]) for k in sorted(data_to_analysize, key=data_to_analysize.get, reverse=True)]
        #sorted_x = sorted(data_to_analysize.items(), key=get(),reverse=True)
        print(list(sorted_x)[:(int(args.top_occurances))])
        value = list(sorted_x)[:(int(args.top_occurances))]
        with open('feature_file'+args.file_name+".txt", 'w') as filewrite:
            for item in list(sorted_x)[:(int(args.top_occurances))]:
                value,num = item
                if num != 0:
                    filewrite.write("%s\n" % value)

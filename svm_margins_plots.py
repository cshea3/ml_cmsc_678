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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name_of_l1_model", help="path to model")
    parser.add_argument("name_of_r1_model")
    #parser.add_argument("name_of_10_model")
    #parser.add_argument("name_of_100_model")
    #parser.add_argument("path_to_label", help="path to labels")
    parser.add_argument("test_data", help="file that holds the test data")
    parser.add_argument("test_labels",help="file that holds the test labels")

    #parser.add_argument("filename", help="prefix file name")
    #parser.add_argument("svm_type", help="linear, poly or rfb")
    args = parser.parse_args()
    X = np.array([])
    Y = np.array([])

    with open(args.name_of_l1_model,'rb') as fid:
        clf = cPickle.load(fid)
    

    #with open(args.name_of_r1_model,'rb') as fid:
    #   svm_r1 = cPickle.load(fid)

    with open(args.test_labels,'rb') as fid:
        Y = cPickle.load(fid)

    with open(args.test_data,'rb') as fid:
        X = cPickle.load(fid)
    
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy + a * margin
    yy_up = yy - a * margin
    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10)
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)

    plt.axis('tight')
    x_min = -4.8
    x_max = 4.2
    y_min = -6
    y_max = 6

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    #print(YY) 
    #print(len(YY))
    #Z = clf.predict(X,Y)

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())

    plt.show()



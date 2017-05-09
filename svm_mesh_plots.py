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
        smv_l1 = cPickle.load(fid)
    

    with open(args.name_of_r1_model,'rb') as fid:
       svm_r1 = cPickle.load(fid)

    with open(args.test_labels,'rb') as fid:
        Y = cPickle.load(fid)

    with open(args.test_data,'rb') as fid:
        X = cPickle.load(fid)
    
#C = 1.0  # SVM regularization parameter
#svc = svm.SVC(kernel='linear', C=C).fit(X, y)
#rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
#lin_svc = svm.LinearSVC(C=C).fit(X, y)
    h=.02
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']

    
    for i, clf in enumerate((svm_l1, svm_r1)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()


    plt.xticks(())
    plt.yticks(())

    plt.show()



print("There are " +str(data) + " data samples")
    print("There are " +str(labels) + " labels samples")

    np.random.seed(0)
    order = np.random.permutation(n_sample)
    data = data[order]
    labels = labels[order].astype(np.float)
    print(labels)

    data_train = data[:.9 * n_sample]
    labels_train = labels[:.9 * n_sample]
    data_test = data[.9 * n_sample:]
    labels_test = labels[.9 * n_sample:]
    with open('dump_of_data_train_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(data_train,fid)
    with open('dump_of_label_train_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(labels_train,fid)
    with open('dump_of_data_test_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(data_test,fid)
    with open('dump_of_label_test_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(label_test,fid)
    #shape_l = labels.shape
    labels_01 = np.zeros(shape=labels.shape)
    np.copyto(labels_01,labels)
    #for the first attempt only look for class 0 and not class 0
    for x in np.nditer(labels_01):
        if x > 0: labels_01[int(x)]=1

    print(labels_01)
    data_train_01 = data[:.9 * n_sample]
    labels_train_01 = labels[:.9 * n_sample]
    data_test_01 = data[.9 * n_sample:]
    labels_test_01 = labels[.9 * n_sample:]

    #begin to trainup the dataset
    # TODO - train SVM using X and Y from 'data'
    clf = svm.SVC(kernel='linear', gamma=10)
    #send the data for training
    clf.fit(data_train_01, labels_train_01)
    #retrive the predicited labesls
    predict_labels_01 = clf.predict(data_test_01)
    with open('dump_of_svm01_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid)
    
    with open('predicted_01'+str(args.filename)+'.pckl','wb') as pckle_f:
        cPickle.dump(predict_labels_01,pckle_f)

    ################## Linear
    print("Entering Linear SVM .....")
    clf = svm.SVC(kernel='linear',class_weight='balanced',C=.1)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_c.1_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid)
    predict_labels = clf.predict(data_test)
    with open('predicted_c.1_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f)
    
    clf = svm.SVC(kernel='linear',class_weight='balanced',C=1)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_c1_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid)
    predict_labels = clf.predict(data_test)
    with open('predicted_c1_'+str(args.filename)+'.pckl','wb') as pckle_f:
        cPickle.dump(predict_labels,pckle_f)


    clf = svm.SVC(kernel='linear',class_weight='balanced',C=10)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_c10_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid)
    predict_labels = clf.predict(data_test)
    with open('predicted_c10_'+str(args.filename)+'.pckl','wb') as pckle_f:
        cPickle.dump(predict_labels,pckle_f)


    clf = svm.SVC(kernel='linear',class_weight='balanced',C=100)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_c100_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid)
    predict_labels = clf.predict(data_test)
    with open('predicted_c100_'+str(args.filename)+'.pckl','wb') as pckle_f:
        cPickle.dump(predict_labels,pckle_f)

    ################## RBF
    print("Entering RBF SVM ....")
    clf = svm.SVC(kernel='rbf',class_weight='balanced',gamma=0.7, C=.1)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_rbf_c.1_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid)
    predict_labels = clf.predict(data_test)
    with open('predicted_rbf_c.1_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f)
 
    clf = svm.SVC(kernel='rbf',class_weight='balanced',gamma=0.7, C=1)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_rbf_c1_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid)
    predict_labels = clf.predict(data_test)
    with open('predicted_rbf_c1_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f)
     
    clf = svm.SVC(kernel='rbf',class_weight='balanced',gamma=0.7, C=10)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_rbf_c10_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid)
    predict_labels = clf.predict(data_test)
    with open('predicted_rbf_c10_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f)

    clf = svm.SVC(kernel='rbf',class_weight='balanced',gamma=0.7, C=100)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_rbf_c100_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid)
    predict_labels = clf.predict(data_test)
    with open('predicted_rbf_c100_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f)
 
    #################### Poly
    print("Entering Poly SVM ...." )
    clf = svm.SVC(kernel='poly',class_weight='balanced',gamma=0.7, C=.1)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_poly_c.1_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid)
    predict_labels = clf.predict(data_test)
    with open('predicted_poly_c.1_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f)
 
    clf = svm.SVC(kernel='poly',class_weight='balanced',gamma=0.7, C=1)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_poly_c1_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid)
    predict_labels = clf.predict(data_test)
    with open('predicted_poly_c1_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f)
     
    clf = svm.SVC(kernel='poly',class_weight='balanced',gamma=0.7, C=10)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_poly_c10_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid)
    predict_labels = clf.predict(data_test)
    with open('predicted_poly_c10_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f)

    clf = svm.SVC(kernel='poly',class_weight='balanced',gamma=0.7, C=100)
    clf.fit(data_train,labels_train)
    with open('dump_of_svm_poly_c100_'+str(args.filename)+'.pkl','wb') as fid:
        cPickle.dump(clf,fid)
    predict_labels = clf.predict(data_test)
    with open('predicted_poly_c100_'+str(args.filename)+'.pckl','wb') as pckle_f: 
        cPickle.dump(predict_labels,pckle_f)

    #plt.title('inear')
    #plt.show()
    #print("Classification report for classifier %s:\n%s\n"
    #  % (clf, metrics.classification_report(labels_test_01, predicted)))
    #print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels_test_01, predicted))

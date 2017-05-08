if (args.svm_type == "linear"):
        clf = svm.SVC(kernel='linear',class_weight='balanced',C=.1)
        clf.fit(data_train,labels_train)
        with open('dump_of_svm_c.1_'+str(args.filename)+'.pkl','wb') as fid:
            cPickle.dump(clf,fid)
        predict_labels = clf.predict(data_test)
        with open('predicted_c.1_'+str(args.filename)+'.pckl','wb') as pckle_f: 
            cPickle.dump(predict_labels,pckle_f, protocol=4)
        print("finished C=.1, starting C=1")
        clf = svm.SVC(kernel='linear',class_weight='balanced',C=1)
        clf.fit(data_train,labels_train)
        with open('dump_of_svm_c1_'+str(args.filename)+'.pkl','wb') as fid:
            cPickle.dump(clf,fid, protocol=4)
        predict_labels = clf.predict(data_test)
        with open('predicted_c1_'+str(args.filename)+'.pckl','wb') as pckle_f:
            cPickle.dump(predict_labels,pckle_f, protocol=4)
        print("finished C=1, starting C=10")
        clf = svm.SVC(kernel='linear',class_weight='balanced',C=10)
        clf.fit(data_train,labels_train)
        with open('dump_of_svm_c10_'+str(args.filename)+'.pkl','wb') as fid:
            cPickle.dump(clf,fid, protocol=4)
        predict_labels = clf.predict(data_test)
        with open('predicted_c10_'+str(args.filename)+'.pckl','wb') as pckle_f:
            cPickle.dump(predict_labels,pckle_f, protocol=4)
        print("finished C=10, starting C=100")

        clf = svm.SVC(kernel='linear',class_weight='balanced',C=100)
        clf.fit(data_train,labels_train)
        with open('dump_of_svm_c100_'+str(args.filename)+'.pkl','wb') as fid:
            cPickle.dump(clf,fid, protocol=4)
        predict_labels = clf.predict(data_test)
        with open('predicted_c100_'+str(args.filename)+'.pckl','wb') as pckle_f:
            cPickle.dump(predict_labels,pckle_f, protocol=4)
    elif (args.svm_type == "rbf"):
        print("Entering RBF SVM ....")
        clf = svm.SVC(kernel='rbf',class_weight='balanced',gamma=0.7, C=.1)
        clf.fit(data_train,labels_train)
        with open('dump_of_svm_rbf_c.1_'+str(args.filename)+'.pkl','wb') as fid:
            cPickle.dump(clf,fid, protocol=4)
        predict_labels = clf.predict(data_test)
        with open('predicted_rbf_c.1_'+str(args.filename)+'.pckl','wb') as pckle_f: 
            cPickle.dump(predict_labels,pckle_f, protocol=4)
        print("finished C=.1, starting C=1")
        clf = svm.SVC(kernel='rbf',class_weight='balanced',gamma=0.7, C=1)
        clf.fit(data_train,labels_train)
        with open('dump_of_svm_rbf_c1_'+str(args.filename)+'.pkl','wb') as fid:
            cPickle.dump(clf,fid, protocol=4)
        predict_labels = clf.predict(data_test)
        with open('predicted_rbf_c1_'+str(args.filename)+'.pckl','wb') as pckle_f: 
            cPickle.dump(predict_labels,pckle_f, protocol=4)
        print("finished C=1, starting C=10")
        clf = svm.SVC(kernel='rbf',class_weight='balanced',gamma=0.7, C=10)
        clf.fit(data_train,labels_train)
        with open('dump_of_svm_rbf_c10_'+str(args.filename)+'.pkl','wb') as fid:
            cPickle.dump(clf,fid, protocol=4)
        predict_labels = clf.predict(data_test)
        with open('predicted_rbf_c10_'+str(args.filename)+'.pckl','wb') as pckle_f: 
            cPickle.dump(predict_labels,pckle_f, protocol=4)
        print("finished C=10, starting C=100")
        clf = svm.SVC(kernel='rbf',class_weight='balanced',gamma=0.7, C=100)
        clf.fit(data_train,labels_train)
        with open('dump_of_svm_rbf_c100_'+str(args.filename)+'.pkl','wb') as fid:
            cPickle.dump(clf,fid, protocol=4)
        predict_labels = clf.predict(data_test)
        with open('predicted_rbf_c100_'+str(args.filename)+'.pckl','wb') as pckle_f: 
            cPickle.dump(predict_labels,pckle_f, protocol=4)
    elif (args.svm_type == "poly"):
        print("Entering Poly SVM ...." )
        clf = svm.SVC(kernel='poly',class_weight='balanced',gamma=0.7, C=.1)
        clf.fit(data_train,labels_train)
        with open('dump_of_svm_poly_c.1_'+str(args.filename)+'.pkl','wb') as fid:
            cPickle.dump(clf,fid, protocol=4)
        predict_labels = clf.predict(data_test)
        with open('predicted_poly_c.1_'+str(args.filename)+'.pckl','wb') as pckle_f: 
            cPickle.dump(predict_labels,pckle_f, protocol=4)
        print("finished C=.1, starting C=1")
        clf = svm.SVC(kernel='poly',class_weight='balanced',gamma=0.7, C=1)
        clf.fit(data_train,labels_train)
        with open('dump_of_svm_poly_c1_'+str(args.filename)+'.pkl','wb') as fid:
            cPickle.dump(clf,fid, protocol=4)
        predict_labels = clf.predict(data_test)
        with open('predicted_poly_c1_'+str(args.filename)+'.pckl','wb') as pckle_f: 
            cPickle.dump(predict_labels,pckle_f, protocol=4)
        print("finished C=1, starting C=10")
        clf = svm.SVC(kernel='poly',class_weight='balanced',gamma=0.7, C=10)
        clf.fit(data_train,labels_train)
        with open('dump_of_svm_poly_c10_'+str(args.filename)+'.pkl','wb') as fid:
            cPickle.dump(clf,fid, protocol=4)
        predict_labels = clf.predict(data_test)
        with open('predicted_poly_c10_'+str(args.filename)+'.pckl','wb') as pckle_f: 
            cPickle.dump(predict_labels,pckle_f, protocol=4)
        print("finished C=10, starting C=100")
        clf = svm.SVC(kernel='poly',class_weight='balanced',gamma=0.7, C=100)
        clf.fit(data_train,labels_train)
        with open('dump_of_svm_poly_c100_'+str(args.filename)+'.pkl','wb') as fid:
            cPickle.dump(clf,fid, protocol=4)
        predict_labels = clf.predict(data_test)
        with open('predicted_poly_c100_'+str(args.filename)+'.pckl','wb') as pckle_f: 
            cPickle.dump(predict_labels,pckle_f, protocol=4)

###Algorithm Comparison
accuracy1=[]   ###Create seven blank list for the accuracy of seven machine learning method
accuracy2=[]
accuracy3=[]
accuracy4=[]
accuracy5=[]
accuracy6=[]
accuracy7=[]

###random state=0 is the best, but it can be changed to 50 since 
###we want to test its robustness

###SVM
for j in range(20,50):
        for k in range(20,50):
            model = svm.SVC(C=j/10,kernel='rbf', gamma=k/10,   ###Cost and gamma can be modified
                        decision_function_shape='ovf',random_state=0) 
            model.fit(x_train, y_train)
            accuracy1.append(model.score(x_test,y_test))   
            y_pred=model.predict(x_test)
            print(confusion_matrix(y_test,y_pred))
            
            
##RF Classifier 
for j in range(100,200):
        for k in range(10,20):
            rfr = RandomForestClassifier(n_estimators= j, max_features = "auto", oob_score = True, 
                              max_depth=k, random_state=50) ###Tree depth and tree numbers can be modified
            rfr.fit(x_train, y_train)
            accuracy2.append(rfr.score(x_test,y_test))
            y_pred=rfr.predict(x_test)
            print(confusion_matrix(y_test,y_pred))


##  Decision Tree            
for j in range(1,2):
        for k in range(10,50):
            clf = tree.DecisionTreeClassifier(max_depth=k) ###Depth can be modified
            clf.fit(x_train, y_train)
            accuracy3.append(clf.score(x_test,y_test))                            
            y_pred=clf.predict(x_test)
            print(confusion_matrix(y_test,y_pred))           
         
 
##  Adaboost          
for j in range(1,2):
        for k in range(100,200):
            ### Tree numbers can be modified
            clf = AdaBoostClassifier(n_estimators=k, random_state=0,learning_rate=0.8)
            clf.fit(x_train, y_train)
            accuracy4.append(clf.score(x_test,y_test))   
            y_pred=clf.predict(x_test)
            print(confusion_matrix(y_test,y_pred))            
            
##  ANN

for j in range(1,2):
        for k in range(1,101):
            ### Hidden layer numbers are confirmed(because we use MLP Classifier) ,
            ### but the parameter numbers on each layer can be modified.
            ANN=MLPClassifier(hidden_layer_sizes=[k], activation='relu', solver='adam', random_state=0)
            ANN.fit(x_train, y_train)
            accuracy5.append(ANN.score(x_test,y_test))   
            y_pred=ANN.predict(x_test)
            print(confusion_matrix(y_test,y_pred))            
            
            
## KNN
for j in range(1,2):
        for k in range(1,101):
            ### The parameter k can be modified
            knn_classifier=KNeighborsClassifier(k)
            knn_classifier.fit(x_train,y_train) 
            accuracy6.append(knn_classifier.score(x_test,y_test)) 
            y_pred=knn_classifier.predict(x_test)
            print(confusion_matrix(y_test,y_pred)) 
            
##GBM
for j in range(10,20):
        for k in range(100,200):
            ###The tree numbers and depth can be modified
            GBM=GradientBoostingClassifier(n_estimators=k,max_depth=j)
            GBM.fit(x_train,y_train) 
            accuracy7.append(GBM.score(x_test,y_test))
            y_pred=GBM.predict(x_test)
            print(confusion_matrix(y_test,y_pred)) 





###Compare the accuracy by showing figure
xxxx=pd.read_excel('accuracy.xlsx')              
algorithms=['SVM','RF','DT ','Adaboost','ANN','KNN','GBM','Bayes','PMV']
xxxx.boxplot() 
plt.xticks(rotation=30,fontsize=10)
plt.title("The accuracy of classification models",fontsize=10)
plt.ylabel("Accuracy",fontsize=10)
plt.xlabel("Algorithms(Classifier)",fontsize=10)
plt.show()            
            
            
            
            
            
            
            
            
            
            
            





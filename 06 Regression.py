###Regression

features = pd.read_excel('Dataset of three categories.xlsx') 
features = pd.read_excel('Dataset of seven categories.xlsx') 

labels = np.array(features['TS'])   
features = features.drop('TS', axis = 1)
feature_list = list(features.columns) 

x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.2)
def encode_features(x_train,x_test):
    features = ['Season','Building','Mode','Sex']
    df_combined = pd.concat([x_train[features],x_test[features]])

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        x_train[feature] = le.transform(x_train[feature])
        x_test[feature] = le.transform(x_test[feature])
    return x_train, x_test

x_train, x_test = encode_features(x_train,x_test)




LR=LinearRegression()
LR.fit(x_train,y_train)
LR_y_pred=LR.predict(x_test)
print('Coefficients: \n', LR.coef_)
print("Mean squared error: %.2f"% mean_squared_error(y_test, LR_y_pred))
print('Variance score: %.2f' % r2_score(y_test, LR_y_pred))
 

accuracy=[]
###Logistic Regression
for k in range(1,100):
    clf=LogisticRegression(C=0.36)
    clf.fit(x_train,y_train)
    clf_y_pred=clf.predict(x_test)
    print(clf.score(x_test,y_test))
    








print(clf.score(x_test,y_test)) 
from sklearn.metrics import confusion_matrix
print(confusion_matrix(clf_y_pred,y_test))
print('Coefficients: \n', clf.coef_)
print("Mean squared error: %.2f"% mean_squared_error(y_test, clf_y_pred))
print('Variance score: %.2f' % r2_score(y_test, clf_y_pred))

###RandomForest Regression
rfr = RandomForestRegressor(n_estimators= 300, max_features = "auto", oob_score = True, 
                              max_depth=40, random_state=50) 
rfr.fit(x_train,y_train)
rfr_y_pred=rfr.predict(x_test)

print("Mean squared error: %.2f"% mean_squared_error(y_test, rfr_y_pred))
print('Variance score: %.2f' % r2_score(y_test, rfr_y_pred))

###DecisionTree Regression
rgr = DecisionTreeRegressor(random_state = 0,max_depth=6)
rgr.fit(x_train,y_train)
rgr_y_pred=rgr.predict(x_test)

print("Mean squared error: %.2f"% mean_squared_error(y_test, rgr_y_pred))
print('score: %.2f' % rgr.score(x_test,y_test))


###SVM Regression

SVM=svm.SVR()
SVM.fit(x_train,y_train)
SVM_y_pred=SVM.predict(x_test)

print("Mean squared error: %.2f"% mean_squared_error(y_test, SVM_y_pred))
print('score: %.2f' % SVM.score(x_test,y_test))






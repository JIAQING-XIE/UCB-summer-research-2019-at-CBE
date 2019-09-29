###Data input and type transformation


###Depends on machine learning on seven categories or three categories
features = pd.read_excel('Dataset of three categories.xlsx') 
features = pd.read_excel('Dataset of seven categories.xlsx') 
 
features = pd.read_excel('ml-1.xlsx') 
labels = np.array(features['TS'])          ###Seperate the labels from features 
features = features.drop('TS', axis = 1)
feature_list = list(features.columns)      ###Gather the features in a column and in a sequence

###Train and test data split and Label encoding
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
def encode_features(x_train,x_test):
    ##Only suitable to the dataset has the features which are not numeric
    features = ['Season','Building','Mode','Sex'] 
    #features = []  Suitable to the dataset has the features which are numeric
    df_combined = pd.concat([x_train[features],x_test[features]])
    for feature in features:
        le = preprocessing.LabelEncoder()  ###Label encoder method
        le = le.fit(df_combined[feature])
        x_train[feature] = le.transform(x_train[feature]) ###The transformation process
        x_test[feature] = le.transform(x_test[feature])
    return x_train, x_test
    
x_train, x_test = encode_features(x_train,x_test) ###Complete the transformation











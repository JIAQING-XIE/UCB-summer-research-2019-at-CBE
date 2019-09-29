###Five Sampling Methods


###In this study, we choose ROS as the oversampling method.
ROS = RandomOverSampler(random_state = 0)       ###OverSampling
x_train, y_train = ROS.fit_sample(x_train, y_train)



SMOTEE = SMOTE(random_state=0)                  ###OverSampling
x_train,y_train = SMOTEE.fit_sample(x_train,y_train) 



enn = EditedNearestNeighbours(random_state=0)   ###DownSampling
x_train,y_train = enn.fit_sample(x_train, y_train)



###Combinatin of OverSampling and DownSampling (Two methods)
smote_enn = SMOTEENN(random_state=0)
x_train,y_train = smote_enn.fit_sample(x_train, y_train)




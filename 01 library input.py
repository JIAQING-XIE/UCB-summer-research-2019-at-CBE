### Import all python libraries
import numpy as np        ###Numpy 
import pandas as pd       ###Pandas 
import matplotlib.pyplot as plt   ##Matplotlib


from sklearn import preprocessing     ### It can import several kinds of data preprocessing methods
from sklearn.preprocessing import StandardScaler     ### Three data preprocessing methods
from sklearn.preprocessing import Normalizer         
from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import train_test_split   ###Split train and test data


from imblearn.over_sampling import SMOTE   ###Three kinds of sampling methods
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours


####Machine learning libraries which is represented as scikit-learn in python
from sklearn import svm                                      ###Support Vector Machine
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor           ###Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier              ###Adaboost
from sklearn import tree                                    ###Decision Tree 


from sklearn.neural_network import MLPClassifier             ###Artificial Nerual Network
from sklearn.neighbors import KNeighborsClassifier           ###k-nearest neighours
from sklearn.ensemble import GradientBoostingClassifier      ###Gradient Boosting Machine
from sklearn.naive_bayes import GaussianNB                   ###Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


from sklearn.metrics import confusion_matrix                 ###To see the classified results

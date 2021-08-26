import yaml
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Naive Bayes Approach
from sklearn.naive_bayes import MultinomialNB
# Trees Approach
from sklearn.tree import DecisionTreeClassifier
# Ensemble Approach
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#KNN Approach
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sn
import pickle


# data path
pathTrain = "./data/train.csv"

pathTest =  "./data/test.csv"

# store data in dataframe
df_train = pd.read_csv(pathTrain)

df_test = pd.read_csv(pathTest)


#print some info

df_train.head(10)
print('Hello')
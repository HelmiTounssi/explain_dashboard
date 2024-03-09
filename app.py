import pandas as pd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score, roc_curve
import logging
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV


import sys
import os

test_data  = pd.read_csv('./saved_models/y_test.csv')
test_data.to_pickle('./saved_models/y_test.pkl')
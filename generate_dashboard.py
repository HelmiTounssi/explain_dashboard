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
import dill

import sys
import os

# Obtenez le chemin absolu du répertoire parent du notebook
notebook_directory = os.path.abspath('')
# Construisez le chemin absolu du répertoire src
# import data.features.fem_data  as fem
# import models.train.Boosting  as boost
# load a source module from a file
from google.cloud import storage
import pandas as pd
import pickle
import gcsfs
project_id = 'elevated-nuance-414716' 
bucket_name = 'data-model-home-credit' 
model_file = 'clf_lightgbm_100_fold_2_model_.pkl'
#train_data_100 = 'x_train.csv'
#test_data_path = 'y_train.csv'
train_data = 'train_data_final.pkl'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'elevated-nuance-414716-43c8fcd49778.json'
# Initialise a client
client = storage.Client(project_id)
# Create a bucket object for our bucket
bucket = client.get_bucket(bucket_name)

#train_data = pd.read_csv('gs://data-model-home-credit/x_test.csv')
#test_data  = pd.read_csv('gs://data-model-home-credit/y_test.csv')
fs = gcsfs.GCSFileSystem()

with fs.open('gs://data-model-home-credit/x_test.pickle', 'rb') as f:
    train_data = pickle.load(f)
with fs.open('gs://data-model-home-credit/y_test.pkl', 'rb') as f:
    test_data = pickle.load(f)    
# Convert the set to a list
#columns_list = list(train_data.columns)
# Now use the list to select the columns from the DataFrame
#df_selected = test_data[columns_list]
#test_data=df_selected

model = None
# 1. Charger le modèle à partir du fichier pickle

with fs.open('gs://data-model-home-credit/lgbm_best_model.pickle', 'rb') as f:
    model = pickle.load(f)
        
# this is so we do not have to drop the column before making predictions
if 'SK_ID_CURR' in train_data.columns:
    train_data = train_data.set_index('SK_ID_CURR')
# this is so we do not have to drop the column before making predictions
if 'SK_ID_CURR' in test_data.columns:
    test_data = test_data.set_index('SK_ID_CURR')  

train_data_10000 = train_data.head(100)
test_data_10000 = test_data.head(100)
      
explainer = ClassifierExplainer(model, train_data_10000, test_data_10000, 
                                # adds a table and hover labels to dashboard
                                labels=['Accepté', 'Refusé'], # defaults to ['0', '1', etc]
                                target = "Not Paid", # defaults to y.name
                                )
with open("explainer.joblib", "wb") as f:
    dill.dump(explainer, f)
db = ExplainerDashboard(explainer, 
                        title="Home Credit Prediction", 
                        model_output='logodds',# defaults to "Model Explainer"
                        shap_interaction=True, # you can switch off tabs with bools
                        )

db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=False)

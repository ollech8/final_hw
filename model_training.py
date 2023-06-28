# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 02:00:32 2023

@author: ASUS VIVOBOOK
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from madlan_data_prep import prepare_data
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import seaborn as sns
from datetime import datetime, timedelta
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn
import subprocess
from madlan_data_prep import prepare_data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer


df = pd.read_excel('output_all_students_Train_v10.xlsx')
df = prepare_data(df)
dft = pd.read_excel('Dataset_for_test.xlsx')
dft = prepare_data(dft)

X_train = df.drop('price', axis=1)
y_train = df['price']
X_test = dft.drop('price', axis=1)
y_test = dft['price']

num_cols = ['room_number', 'Area', 'floor']
cat_cols = ['hasElevator', 'hasParking', 'hasStorage', 'hasBars', 'hasAirCondition',
            'hasBalcony', 'hasMamad', 'handicapFriendly', 'City', 'type', 'condition', 'entranceDate']

label_encoder = LabelEncoder()
for column in cat_cols:
    X_train[column] = label_encoder.fit_transform(X_train[column])
    X_test[column] = label_encoder.transform(X_test[column])



# fill missing values in columns
cat_pipeline = Pipeline([
    ('one_hot_encoding', OneHotEncoder(sparse=False, handle_unknown='ignore', drop='if_binary'))
])
num_pipeline = Pipeline([
    ('numerical_imputation', SimpleImputer(strategy='median')),
    ('scaling', StandardScaler())
])

column_transformer = ColumnTransformer([
    ('numerical_preprocessing', num_pipeline, num_cols),
    ('categorical_preprocessing', cat_pipeline, cat_cols)
], remainder='drop')

preprocessing_model = Pipeline([
    ('preprocessing_step', column_transformer),
    ('model', ElasticNetCV(cv=10, max_iter=1000, l1_ratio=[.1, .3 , .5, .7, .9, .95, .99, 1],
                           alphas=np.logspace(-5, 1, 10)))
])

crv= sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=13)
crv_scores = cross_val_score(preprocessing_model, X_train, y_train, cv=crv, scoring="neg_mean_squared_error")
kfold_mse = np.abs(crv_scores.mean())
kfold_std = crv_scores.std()
print(f"MSE KFold: {np.round(kfold_mse, 1)} kFold std {kfold_std}")

# Fit the pipeline to the training data
preprocessing_model.fit(X_train, y_train)

y_pred = preprocessing_model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, explained_variance_score

def score_model(y_test, y_pred, model_name):
    RMSE = mean_squared_error(y_test, y_pred, squared=False)
    MSE = mean_squared_error(y_test, y_pred)
    R_squared = r2_score(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    MedAE = median_absolute_error(y_test, y_pred)
    EVS = explained_variance_score(y_test, y_pred)
    print(f"Model: {model_name}")
    print(f"RMSE: {np.round(RMSE, 2)}")
    print(f"MSE: {np.round(MSE, 2)}")
    print(f"R-Squared: {np.round(R_squared, 2)}")
    print(f"Mean Absolute Error: {np.round(MAE, 2)}")
    print(f"Median Absolute Error: {np.round(MedAE, 2)}")
    print(f"Explained Variance Score: {np.round(EVS, 2)}")

score_model(y_test, y_pred, "linear_model.ElasticNetCV")

# Save the model to a file
joblib.dump(preprocessing_model, 'trained_model.pkl')

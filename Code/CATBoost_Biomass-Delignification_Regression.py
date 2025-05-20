import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import seaborn as sns
import urllib
import requests
import zipfile
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from skmatter.preprocessing import StandardFlexibleScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy import sparse
from pandas.plotting import scatter_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
from io import StringIO
import catboost
import warnings
warnings.filterwarnings('ignore')

# Import your dataset using pandas
input_dir = 'Path_of_dataset/'
df = pd.read_csv(input_dir + 'Delignification_Organosolv_Literature.csv', encoding= 'unicode_escape')

# Identify features to be scaled with StandardScaler and MinMaxScaler
standard_columns_True = ['Total HSP', 'Cellulose, %', 'Lignin, %', 'Hemicellulose, %', 'Solvent vol. %', 'Solvent viscosity (mPa.s)',
                        'Water vol. %', 'T (oC)', 'time (min)', 'Solid loading (g/mL)', 'Catalyst density (g/mL)',
                        'Catalyst mol-wt (g/mol)', 'pH of catalyst', 'Catalyst loading (mM)']
standard_columns_False = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']

# Split the data into features and target (adjust 'target_column' accordingly)
X = df[standard_columns_True + standard_columns_False]
y = df['Delignification,%']

# Create transformers for each type of scaling
scaler_SFS_True = StandardFlexibleScaler(column_wise=True)
scaler_SFS_False = StandardFlexibleScaler(column_wise=False)

# Create the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('standard_scaler', scaler_SFS_True, standard_columns_True),
        ('standard_scaler_COSMO', scaler_SFS_False, standard_columns_False)], 
    remainder='passthrough')

# Create the pipeline
pipeline = Pipeline([('preprocessor', preprocessor)])

# Fit and transform the data
X_scaled = pipeline.fit_transform(X)

# Convert the result back to a DataFrame
columns = standard_columns_True + standard_columns_False
X_scaled_df = pd.DataFrame(X_scaled, columns=columns)

# Splitting the data into Training and Testing random_state=412, 12412
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.15, random_state=435)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# https://docs.aws.amazon.com/sagemaker/latest/dg/catboost-hyperparameters.html
# pip install catboost
CatBoost_Model = CatBoostRegressor(rsm=0.5, l2_leaf_reg=1)
CatBoost_Model.fit(X_train, y_train, verbose=1000)

# Make prediction for Training
pred_train = CatBoost_Model.predict(X_train)

# Save y_train and y_train_pred to a single CSV file
results_train = pd.DataFrame({
    'y_train_exp': y_train,
    'y_train_pred': pred_train.flatten()
})

#results_train.to_csv('MLR_pred-DES-alpha-Train.csv', index=True)

# Mean absolute error (MAE)
mae_train = mean_absolute_error(y_train.values.ravel(), pred_train)

# Mean squared error (MSE)
mse_train = mean_squared_error(y_train.values.ravel(), pred_train)
rmse_train = (mse_train**0.5)

# mean absolute percentage error (MAPE)
mape_train = mean_absolute_percentage_error(y_train.values.ravel(), pred_train)

# R-squared scores
r2_train = r2_score(y_train.values.ravel(), pred_train)

# Print metrics
print("")
print('R2_Training:', round(r2_train, 3))
print('MAPE_Training:', "{:.2%}".format(mape_train))
print('MAE_Training:', round(mae_train, 2))
print('MSE_Training:', round(mse_train, 2))
print('RMSE_Training:', round(rmse_train, 2))

# Make prediction for Testing
pred_test = CatBoost_Model.predict(X_test)

# Save y_train and y_train_pred to a single CSV file
results_train = pd.DataFrame({
    'y_test_exp': y_test,
    'y_test_pred': pred_test.flatten()
})

#results_train.to_csv('MLR_pred-DES-alpha-Test.csv', index=True)

# Mean absolute error (MAE)
mae_test = mean_absolute_error(y_test.values.ravel(), pred_test)

# Mean squared error (MSE)
mse_test = mean_squared_error(y_test.values.ravel(), pred_test)
rmse_test = (mse_test**0.5)

# mean absolute percentage error (MAPE)
mape_test = mean_absolute_percentage_error(y_test.values.ravel(), pred_test)

# R-squared scores
r2_test = r2_score(y_test.values.ravel(), pred_test)

# Print metrics
print("")
print('R2_Testing:', round(r2_test, 3))
print('MAPE_Testing:', "{:.2%}".format(mape_test))
print('MAE_Testing:', round(mae_test, 2))
print('MSE_Testing:', round(mse_test, 2))
print('RMSE_Testing:', round(rmse_test, 2))

# Define x axis
x_axis_train = y_train
x_axis_test = y_test

plt.figure(figsize=(6,7))

plt.scatter(x_axis_train, pred_train, c = 'b', alpha = 0.8, marker=MarkerStyle("D", fillstyle="right"), s=30, label = 'Training')
plt.scatter(x_axis_test, pred_test, c = 'r', alpha = 0.8, marker=MarkerStyle("D", fillstyle="left"), s=60, label = 'Testing')
plt.xlabel(r'$ln (\eta)$-experimental')
plt.ylabel(r'$ln (\eta)$-predicted')

plt.xticks(np.arange(0, 101, step=10))
plt.yticks(np.arange(0, 101, step=10))

plt.grid(color = '#D3D3D3', linestyle = '--', which='both', axis='both')
plt.legend(loc = 'upper left')
plt.title('CatBoost Predictions with Sigma Descriptors')
plt.show()

#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# In[7]:


# Load your file
df = pd.read_csv("/Users/mandyliao/Downloads/glucose data - s2.csv")

# check if it loaded properly

print(df.head())


## process df

# drop first 2 col
df=df.drop(['Unnamed: 0', 'subject','bp high (equipment)','bp low (eq)','bp high (app)','bp low (app)','respiratory rate', 'RED avg'], axis=1)

#drop empty rows
df=df.dropna()

#change gender into boolean (M=T; F=F)
df['gender']=df['gender'].map({'M':True,'F':False})
df['gender']=df['gender'].astype(int)


#test train split
X=df.drop(columns=['body glucose (mg/dL)'])
y=df['body glucose (mg/dL)']

Xtr, Xts, ytr, yts=train_test_split(X,y,test_size=0.2, random_state=42)


# In[13]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Load your CSV file
df = pd.read_csv("/Users/mandyliao/Downloads/glucose data - s2.csv")

# Preprocess the data
# Drop unnecessary columns
df = df.drop(['Unnamed: 0', 'subject', 'bp high (equipment)', 'bp low (eq)', 
              'bp high (app)', 'bp low (app)', 'respiratory rate', 'RED avg'], axis=1)

# Drop any empty rows
df = df.dropna()

# Convert gender to boolean (M=True; F=False) and then to int
df['gender'] = df['gender'].map({'M': True, 'F': False}).astype(int)

# Define features and target
X = df.drop(columns=['body glucose (mg/dL)'])
y = df['body glucose (mg/dL)']

# Split the dataset into training and testing sets
Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up PolynomialFeatures with degree 3 to get individual, pairwise, and triple interactions
poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)

# Transform the training and test sets to include interactions
Xtr_poly = poly.fit_transform(Xtr)
Xts_poly = poly.transform(Xts)

# Get the feature names after polynomial transformation
feature_names = poly.get_feature_names_out(X.columns)

# Display feature names after transformation (for verification)
print("Feature names after polynomial transformation:\n", feature_names)

# Set up a pipeline with StandardScaler and RandomForestRegressor
pipeline = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42))

# Perform cross-validation on the training set
cv_scores = cross_val_score(pipeline, Xtr_poly, ytr, cv=5, scoring='neg_mean_absolute_error')
print("Cross-validation MAE (training set):", -np.mean(cv_scores))

# Fit the model on the training set
pipeline.fit(Xtr_poly, ytr)

# Make predictions on the test set
y_pred = pipeline.predict(Xts_poly)

# Calculate performance metrics
mae = mean_absolute_error(yts, y_pred)
mse = mean_squared_error(yts, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(yts, y_pred)

print("Mean Absolute Error (Test Set):", mae)
print("Mean Squared Error (Test Set):", mse)
print("Root Mean Squared Error (Test Set):", rmse)
print("R2 Score:", r2)

# Calculate feature importance using permutation importance
results = permutation_importance(pipeline, Xts_poly, yts, scoring='neg_mean_absolute_error')

# Create a DataFrame to display feature importance with correct feature names
feature_importance = pd.DataFrame({'feature': feature_names, 'importance': results.importances_mean})

results = permutation_importance(pipeline, Xts_poly, yts, scoring='neg_mean_absolute_error')
print("Feature Importance:\n", feature_importance.sort_values(by='importance', ascending=False))

# Filter features with importance greater than 10
important_features = feature_importance[feature_importance['importance'] > 0]

# Display important features
print("Features with Importance Greater Than threshold:\n", important_features)

# If there are important features, update the feature set
if not important_features.empty:
    # Create a mask for the features to keep
    mask = np.isin(feature_names, important_features['feature'].values)
    # Select only the important features from the original polynomial transformed data
    Xtr_poly_filtered = Xtr_poly[:, mask]
    Xts_poly_filtered = Xts_poly[:, mask]
    
    # Fit the model again on the filtered training set
    pipeline.fit(Xtr_poly_filtered, ytr)
    
    # Make predictions on the filtered test set
    y_pred_filtered = pipeline.predict(Xts_poly_filtered)

    # Calculate performance metrics for the filtered model
    mae_filtered = mean_absolute_error(yts, y_pred_filtered)
    mse_filtered = mean_squared_error(yts, y_pred_filtered)
    rmse_filtered = np.sqrt(mse_filtered)
    r2_filtered = r2_score(yts, y_pred_filtered)

    mard = np.mean(np.abs((yts - y_pred_filtered) / yts))

    print("Random Forest Regressor")
    print("Mean Absolute Relative Deviation (MARD):", mard)
    print("Mean Absolute Error:", mae_filtered)
    print("Mean Squared Error:", mse_filtered)
    print("Root Mean Squared Error:", rmse_filtered)
    print("R2 Score:", r2_filtered)
else:
    print("No features have importance greater than 10.")


# In[ ]:





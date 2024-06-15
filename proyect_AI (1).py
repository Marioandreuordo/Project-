#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('train.csv')  # Adjust the path to your dataset file

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nChecking for missing values:")
print(data.isnull().sum())

# Split the data into features and target
X = data.drop('price_range', axis=1)
y = data['price_range']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural Network Implementation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialize the Neural Network model
nn_model = Sequential()
nn_model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
nn_model.add(Dense(16, activation='relu'))
nn_model.add(Dense(8, activation='relu'))
nn_model.add(Dense(4, activation='softmax'))  # Assuming 4 price ranges

# Compile the model
nn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
nn_model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

# Make predictions
nn_y_pred = np.argmax(nn_model.predict(X_test), axis=-1)

# Evaluate the model
nn_accuracy = accuracy_score(y_test, nn_y_pred)
print(f'Neural Network Accuracy: {nn_accuracy}')
print('Classification Report for Neural Network:')
print(classification_report(y_test, nn_y_pred))

# Confusion matrix for Neural Network
nn_cm = confusion_matrix(y_test, nn_y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(nn_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Neural Network')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Random Forest Implementation
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
rf_y_pred = rf_model.predict(X_test)

# Evaluate the model
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f'Random Forest Accuracy: {rf_accuracy}')
print('Classification Report for Random Forest:')
print(classification_report(y_test, rf_y_pred))

# Confusion matrix for Random Forest
rf_cm = confusion_matrix(y_test, rf_y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Cargar el dataset de entrenamiento
train_data = pd.read_csv('train.csv')

# Inspeccionar los datos de entrenamiento
print(train_data.head())
print(train_data.info())
print(train_data.describe())

# Preprocesamiento de datos
# Manejar valores faltantes si los hubiera
train_data.fillna(train_data.mean(), inplace=True)

# Selección de todas las características y la variable objetivo
X = train_data.drop('price_range', axis=1)
y = train_data['price_range']

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creación y entrenamiento del modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('Modelo con todas las características:')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Visualización de los resultados
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.show()


# In[8]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('train.csv')  # Adjust the path to your dataset file

# Inspect the data
print("First few rows of the dataset:")
print(data.head())
print("\nDataset information:")
print(data.info())
print("\nDataset statistical description:")
print(data.describe())

# Data preprocessing
# Handle missing values if any
data.fillna(data.mean(), inplace=True)

# Feature selection for PCA (excluding 'price_range' column as it is the target variable)
features = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 
            'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 
            'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 
            'touch_screen', 'wifi']
X = data[features]
y = data['price_range']

# Data normalization
X_scaled = StandardScaler().fit_transform(X)

# Function to evaluate PCA with different numbers of components
def evaluate_pca(n_components):
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(X_scaled)
    principalDf = pd.DataFrame(data=principalComponents, 
                               columns=[f'principal_component_{i+1}' for i in range(n_components)])
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(principalDf, y, test_size=0.2, random_state=42)
    
    model_pca = LinearRegression()
    model_pca.fit(X_train_pca, y_train_pca)
    y_pred_pca = model_pca.predict(X_test_pca)
    
    mse = mean_squared_error(y_test_pca, y_pred_pca)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_pca, y_pred_pca)
    
    return mse, rmse, r2, pca.components_

# Evaluate PCA with all possible numbers of components
results = []
for n in range(1, len(features) + 1):
    mse, rmse, r2, components = evaluate_pca(n)
    results.append((n, mse, rmse, r2, components))
    print(f'Model with {n} principal components:')
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R-squared: {r2}')
    print(f'Selected principal components:\n{components}\n')

# Find the best model based on R-squared
best_result = max(results, key=lambda x: x[3])
best_n_components = best_result[0]
best_components = best_result[4]

print(f'\nBest model: {best_n_components} principal components')
print(f'Mean Squared Error: {best_result[1]}')
print(f'Root Mean Squared Error: {best_result[2]}')
print(f'R-squared: {best_result[3]}')
print(f'Selected principal components:\n{best_components}')

# Visualization of results
plt.figure(figsize=(14, 6))

# Results of the model with the best PCA
pca = PCA(n_components=best_n_components)
principalComponents = pca.fit_transform(X_scaled)
principalDf = pd.DataFrame(data=principalComponents, 
                           columns=[f'principal_component_{i+1}' for i in range(best_n_components)])
X_train_best_pca, X_test_best_pca, y_train_best_pca, y_test_best_pca = train_test_split(principalDf, y, test_size=0.2, random_state=42)
model_best_pca = LinearRegression()
model_best_pca.fit(X_train_best_pca, y_train_best_pca)
y_pred_best_pca = model_best_pca.predict(X_test_best_pca)

plt.subplot(1, 2, 1)
plt.scatter(y_test_best_pca, y_pred_best_pca)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(f'True Values vs Predictions (PCA with {best_n_components} components)')

# Results of the model with original features
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)
model_orig = LinearRegression()
model_orig.fit(X_train_orig, y_train_orig)
y_pred_orig = model_orig.predict(X_test_orig)

plt.subplot(1, 2, 2)
plt.scatter(y_test_orig, y_pred_orig)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions (Original)')

plt.tight_layout()
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Cargar el dataset
data = pd.read_csv('test.csv')

# Inspeccionar los datos
print(data.head())
print(data.info())
print(data.describe())

# Preprocesamiento de datos
# Manejar valores faltantes si los hubiera
data.fillna(data.mean(), inplace=True)

# Selección de características y variable objetivo
features = ['battery_power', 'int_memory', 'ram', 'px_height', 'px_width']
X = data[features]
y = data['price_range']

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

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Visualización de los resultados
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.show()


# In[4]:


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

# Selección de características y variable objetivo
features = ['battery_power', 'int_memory', 'ram', 'px_height', 'px_width']
X = train_data[features]
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

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Visualización de los resultados
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.show()


# In[3]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cargar el dataset
data = pd.read_csv('test.csv')

# Inspeccionar los datos
print(data.head())
print(data.info())
print(data.describe())

# Preprocesamiento de datos
# Manejar valores faltantes si los hubiera (aquí no hay valores faltantes según la descripción)
data.fillna(data.mean(), inplace=True)

# Selección de características para PCA (excluyendo la columna 'id' ya que no es relevante)
features = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 
            'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 
            'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 
            'touch_screen', 'wifi']
X = data[features]

# Normalización de los datos
X = StandardScaler().fit_transform(X)

# Realizar PCA
pca = PCA(n_components=2)  # Aquí reducimos a 2 componentes principales
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal_component_1', 'principal_component_2'])

# Añadir la columna 'id' para referencia
finalDf = pd.concat([principalDf, data[['id']]], axis=1)

# Visualización de los componentes principales
plt.figure(figsize=(8, 6))
plt.scatter(principalDf['principal_component_1'], principalDf['principal_component_2'], s=50)
plt.title('PCA of Mobile Phone Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()

# Mostrar el dataframe con las componentes principales
print(finalDf.head())


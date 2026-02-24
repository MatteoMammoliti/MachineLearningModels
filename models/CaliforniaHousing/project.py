"""
California Housing dataset
--------------------------

:Number of Instances: 20640
:Number of Attributes: 8 numeric, predictive attributes and the target

:Attribute Information:
    - MedInc        median income in block group
    - HouseAge      median house age in block group
    - AveRooms      average number of rooms per household
    - AveBedrms     average number of bedrooms per household
    - Population    block group population
    - AveOccup      average number of household members
    - Latitude      block group latitude
    - Longitude     block group longitude

:Missing Attribute Values: None

The target variable is the median house value for California districts,
expressed in hundreds of thousands of dollars ($100,000).

A block group is the smallest geographical unit for which the U.S.
Census Bureau publishes sample data (a block group typically has a population
of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average
number of rooms and bedrooms in this dataset are provided per household, these
columns may take surprisingly large values for block groups with few households
and many empty houses, such as vacation resorts.
"""

from sklearn.datasets import fetch_california_housing
import pandas as pd
from xgboost import XGBRegressor

pd.set_option("display.max_columns", None)

data = fetch_california_housing()
df = pd.DataFrame(data=data["data"], columns=data["feature_names"])

df["MedValue"] = data["target"]

print(df['MedValue'].describe())

# --- STEP 1: DATA ANALYSIS AND PREPROCESSING
import seaborn as sns
import matplotlib.pyplot as plt

"""
Da questo scatterplot notiamo subito due cose:
- all'aumentare del guadagno medio della zona, aumenta il prezzo medio delle case
- esiste un tetto massimo imposto dal dataset di 500.000 dollari. 
    Consideriamo di rimuovere tali righe per evitare di confondere il modello  
"""
df.plot.scatter(y="MedValue", x="MedInc", alpha=0.5)
plt.show()

"""
Dall'heatmap notiamo come vi è una forte correlazione fra
il valore medio di una casa ed il guadagno medio della zona in cui si trova.
Sono marginali, invece, ai fini della valutazione tutte le altre variabili
poichè vicine o addirittura sotto allo 0. 

Notiamo inoltre che vi sia una correlazione lineare fra il numero delle camere da letto
ed il numero di camere della casa. Ciò significa che l'informazione è ridontante 
non essendo direttamente correlate con il target. Si potrebbe considerare la rimozione di una di esse. 
"""
sns.heatmap(df.corr(), annot=True)
plt.show()

"""
Eliminiamo la colonna delle camere da letto 
e quelle righe dove la valutazione è al limite massimo imposto dal dataset. 
Successivamente effettuiamo lo scaling dei dati poichè l'ordine di grandezza
fra le features è non lineare. 
"""

df.drop("AveBedrms", inplace=True, axis=1)
df.drop(df[df["MedValue"] >= 5].index, inplace=True)
df.plot.scatter(y="MedValue", x="MedInc", alpha=0.5)
plt.show()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop("MedValue", axis=1))
df_numpy = scaler.transform(df.drop("MedValue", axis=1))

features = pd.DataFrame(data=df_numpy, columns=df.columns[:-1])

# --- STEP 2: TRAIN/TEST SPLIT

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import xgboost as xgb

X_train, X_test, y_train, y_test = train_test_split(
    features, df["MedValue"], test_size=0.3, random_state=42
)

# Params per la GridSearch
params = {
    "SVR": {
        "C": [0.1, 1, 10],
        "gamma": [1, 0.1, 0.01],
        "kernel": ["rbf"],
    },
    "XGBRegressor": {
        "n_estimators": [100, 300],
        "max_depth": [5, 10, 15],
        # 'learning_rate': [0.1, 0.01, 0.001],
        # 'gamma': [0, 0.1, 0.01],
        # 'reg_lambda': [1, 10],
        # 'reg_alpha': [0, 0.1]
    },
    "RandomForestRegressor": {
        "n_estimators": [100, 500],
        "max_depth": [5, 10, 15],
    },
}

# Elbow method per il KNN
import numpy as np

results = []

for i in range(1, 40):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    results.append(np.sqrt(mean_squared_error(y_test, pred_i)))

plt.plot(range(1, 40), results)
plt.show()

"""
Dal grafico notiamo subito come:
- tra K=0 e K=5 l'errore scende precipitosamente
- tra K=5 e K=10 l'errore si assesta scendendo di poco
- dopo K=10 le prestazioni peggiorano (rischio di leggero di underfitting)

Considerando che ricerchiamo sempre un trade-off tra affidabilità e semplificità
del modello, scegliamo K = 7
"""
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    KNeighborsRegressor(n_neighbors=7),
]

needs_grid = [RandomForestRegressor(), xgb.XGBRegressor(), SVR()]

# -- STEP 3: MODEL TRAINING AND PREDICTIONS

predictions = {}

for model in models:
    print(f"Fitting {model.__class__.__name__}")
    model.fit(X_train, y_train)
    predictions[model.__class__.__name__] = model.predict(X_test)

for model in needs_grid:
    print(f"Fitting {model.__class__.__name__}")
    grid = GridSearchCV(
        model,
        param_grid=params[model.__class__.__name__],
        refit=True,
        cv=3,
        n_jobs=1,
        verbose=3,
    )
    grid.fit(X_train, y_train)
    predictions[model.__class__.__name__] = grid.predict(X_test)

# -- STEP 4: VALUTAZIONE

for model_name, pred in predictions.items():
    print(f"Valutazione {model_name}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)) * 100.000}")

"""
Output:
Valutazione LinearRegression RMSE: 0.6488119082112033
Valutazione DecisionTreeRegressor RMSE: 0.6443917656943565
Valutazione KNeighborsRegressor RMSE: 0.5811679941351795
Valutazione RandomForestRegressor RMSE: 0.46606903645099085
Valutazione XGBRegressor RMSE: 0.42839628048058404
Valutazione SVR RMSE: 0.511186696649625

Il modello più performante è in assoluto il XGBoost che 
ha un margine di errore di 43.000 dollari. 
Considerando che i prezzi si aggirano in un un range tra 14.000 e
500.000 dollari, il risultato è accettabile. 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- STEP 1: SCALING DATA
pd.set_option("display.max_columns", None)

df = pd.read_csv('./KNN_Project_Data.csv')

print(df.head())



scaler = StandardScaler()

# Let's fit the scaler model
# We remove the 'TARGET CLASS' from the dataset as it shouldn't be scaled
scaler.fit(df.drop('TARGET CLASS', axis=1))

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))

df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])

print(df_scaled.head())

# --- STEP 2: TRAIN/SPLIT
X = df_scaled
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# --- STEP 4: CHOOSING K
error = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error.append(np.mean(pred != y_test))

plot = plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.show()

"""
Looking at the plot (Elbow Method), we can observe three main phases:
- From k=1 to k=6, the error rate strongly decreases.
- From k=7 to k=20, the error rate fluctuates (small ups and downs).
- From k=21 to k=32, it essentially plateaus.

Conclusion: 
A value like k=7 or k=9 seems to be the optimal choice. It is the first point where we achieve a low and stable error rate. Choosing a much higher value (like k=20) would not significantly improve our accuracy, but it would introduce unnecessary computational overhead and risk oversmoothing our model.
"""

# --- STEP 5: PREDICTION AND METRICS
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print(classification_report(y_test, pred))

"""
Output:
              precision    recall  f1-score   support

           0       0.84      0.77      0.80       159
           1       0.76      0.83      0.80       141

    accuracy                           0.80       300
   macro avg       0.80      0.80      0.80       300
weighted avg       0.80      0.80      0.80       300

After testing, we obtained "fair" or "decent" results, with an overall Accuracy of 80% and F1-scores of 0.80 
across the board. With a Precision of 0.76, this model generates many more False Positives. 
"""
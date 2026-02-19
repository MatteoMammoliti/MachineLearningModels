"""
We will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement.
We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.

This data set contains the following features:

* 'Daily Time Spent on Site': consumer time on site in minutes
* 'Age': cutomer age in years
* 'Area Income': Avg. Income of geographical area of consumer
* 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
* 'Ad Topic Line': Headline of the advertisement
* 'City': City of consumer
* 'Male': Whether or not consumer was male
* 'Country': Country of consumer
* 'Timestamp': Time at which consumer clicked on Ad or closed window
* 'Clicked on Ad': 0 or 1 indicated clicking on Ad
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- STEP 1: DATA ANALYSIS
pd.set_option("display.max_columns", None)

advertising = pd.read_csv("./advertising.csv")

# print(advertising.head())
# print(advertising.describe())
# print(advertising.info())

# As we can see from this heatmap, the dataset doesn't contain any NaN cell
heatmap = sns.heatmap(data=advertising.isnull())
plt.show()

"""
The dataset is perfectly balanced: there is an equal number of users who clicked on the Ad and those who didn't (exactly 500 per category).
Analyzing the age distribution, it is interesting to note that:
- Users over 40 are much more likely to click on the ad.
- Users between 25 and 40 show mixed behavior, but with a strong tendency NOT to click.
"""
histplot = sns.histplot(
    data=advertising, x="Age", hue="Clicked on Ad", bins=30, multiple="dodge"
)
plt.show()

"""
This jointplot seems to indicate the presence of a correlation beetween the time spent on the website and
the daily internet usage. Infact, who has an high daily internet usage are likely to spend more
time on the website as well
"""
jointplot_1 = sns.jointplot(
    data=advertising, x="Daily Internet Usage", y="Daily Time Spent on Site", kind="hex"
)
plt.show()

"""
This jointplot reveals the most crucial business insight of the entire dataset: 
the users are cleanly split into two clusters. The users who spend 
less time overall on the internet and on the site are the ones most likely to click the ad, while 
who spend more time consistently ignore the ad.
"""
jointplot_2 = sns.jointplot(
    data=advertising,
    x="Daily Internet Usage",
    y="Daily Time Spent on Site",
    hue="Clicked on Ad",
)
plt.show()

# --- STEP 2: TRAIN/SPLIT

X = advertising[
    ["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Male"]
]

y = advertising["Clicked on Ad"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -- STEP 3: MODEL TRAIN AND PREDICTION

lm = LogisticRegression(max_iter=100)
lm.fit(X_train, y_train)

print(lm.coef_)
"""
Output: [[-0.0557,  0.2661, -0.000017, -0.0273,  0.0023]]

Assuming all other variables remain constant, the model interpreted the coefficients as changes in the log-odds:
- As the 'Daily Time Spent on Site' increases by one minute, the log-odds of clicking the ad decrease by 0.055. 
    In practical terms: spending more time on the site lowers the likelihood of clicking the ad.
- As 'Age' increases by one year, the log-odds increase by 0.266, making older users significantly more likely to click.
"""

prediction = lm.predict(X_test)

report = classification_report(y_test, prediction)

print(report)
"""
              precision    recall  f1-score   support

           0       0.85      0.96      0.90       146
           1       0.96      0.84      0.89       154

    accuracy                           0.90       300
   macro avg       0.90      0.90      0.90       300
weighted avg       0.90      0.90      0.90       300

- Looking at the non-clickers (Class 0), out of 146 actual cases, 
    the model successfully identified 96% of them (Recall = 0.96). This is an excellent result.
- For the users who actually clicked (Class 1), the model is a bit more conservative, correctly identifying 84% of them (Recall = 0.84) out of 154 cases. 
    This means we generated about 16% False Negatives (users who would have clicked, but the model predicted otherwise).
- Interestingly, the model has a very high Precision for Class 1 (0.96). 
    From a business perspective, this is highly valuable: when the model predicts a user will click, it is correct 96% of the time, 
    meaning the company won't waste advertising budget on uninterested users.
- Overall, on the test set of 300 users, the model achieved a total Accuracy of 90% and F1-Score of 90%
    which means that the model is extremely balanced on its predictions.
"""
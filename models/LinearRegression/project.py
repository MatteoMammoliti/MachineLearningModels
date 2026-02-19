"""
You just got some contract work with an Ecommerce company based in New York City that sells clothing online but
they also have in-store style and clothing advice sessions. Customers come in to the store,
have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.

The company is trying to decide whether to focus their efforts on their mobile app experience or their website.
They've hired you on contract to help them figure it out! Let's get started!
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)

customer_file = pd.read_csv("./Ecommerce_Customers.csv")

# --- STEP 1: Data preprocessing and first analysis

# let's remove the non-numerical values we can't currently threat
cleaned_customer_file = customer_file[
    [
        "Avg. Session Length",
        "Time on App",
        "Time on Website",
        "Length of Membership",
        "Yearly Amount Spent",
    ]
]

# print(cleaned_customer_file.head())
# print(cleaned_customer_file.info())
# print(cleaned_customer_file.describe())

"""
We can easly see that there's no clear correlation between the time 
spent on website and the yearly amount spent by customers. 
The statement "The more time you spend on website, the more money you spend" is actually wrong.
"""
joinplot_1 = sns.jointplot(
    x="Time on Website", y="Yearly Amount Spent", data=cleaned_customer_file
)
plt.show()

"""
This plot is much more interesting. There's visible a correlation between the time spent on mobile 
application and the amount of dollars customers spent each year. 
This mean the statement "The more time you spend on app, the more money you spend" is supported by data. 
"""
joinplot_2 = sns.jointplot(
    x="Time on App", y="Yearly Amount Spent", data=cleaned_customer_file
)
plt.show()

"""
In this plot, we can see instead that the length of membership doesn't affect the time spent on the app: 
long-time customers use it on average just as much as new ones.
"""
joinplot_3 = sns.jointplot(
    x="Time on App", y="Length of Membership", data=cleaned_customer_file, kind="hex"
)
plt.show()

"""
From the lmplot, we immediately notice how well the Linear Regression algorithm fits our dataset. 
This means there is a strong linear relationship between the length of membership and the amount spent. 
We can deduce that as customer loyalty increases, the yearly amount spent increases mathematically.
"""
lnplot_1 = sns.lmplot(
    x="Length of Membership", y="Yearly Amount Spent", data=cleaned_customer_file
)
plt.show()

# --- STEP 2: Splitting features and target

"""
We want our model to predict the yearly amount spent by customers 
considering how much time they spent on website or mobile app
"""
y = cleaned_customer_file["Yearly Amount Spent"]
X = cleaned_customer_file[
    [
        "Avg. Session Length",
        "Time on App",
        "Time on Website",
        "Length of Membership",
    ]
]

# --- STEP 3: train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101
)

# --- STEP 4: model train session

lm = LinearRegression()
lm.fit(X_train, y_train)

# Let's visualize the coefficients the model has assigned to all the numerical values
print(lm.coef_)
"""
Output: [25.98154972 38.59015875  0.19040528 61.27909654]
Assuming that all other values remain constant, the model indicates that each 
time the Average Session Length increases by just one minute, 
the yearly amount spent increases by roughly 26 dollars.

The key finding is that when a customer uses the mobile application for one additional minute, 
the e-commerce company gains about 39 dollars. 
On the other hand, the time spent on the website doesn't seem to be a dominant factor (just 0.20 dollars per extra minute)
"""

# --- STEP 5: prediction and metrics
prediction = lm.predict(X_test)

print(f"MAE: {metrics.mean_absolute_error(y_test, prediction)}")
print(f"MSE: {metrics.mean_squared_error(y_test, prediction)}")
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_test, prediction))}")

"""
Final considerations:
As our model got an RMSE of 8.9, which means that its average error is 8.9 dollars, 
and considering the high values we find in the "Yearly Amount Spent" (about 400-500 dollars),
we can consider the results accurate and the project successfully completed. 
"""

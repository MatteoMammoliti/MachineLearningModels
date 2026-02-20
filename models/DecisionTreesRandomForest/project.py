"""
For this project we will be exploring publicly available data from [LendingClub.com](www.lendingclub.com).
Lending Club connects people who need money (borrowers) with people who have money (investors).
Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back.
We will try to create a model that will help predict this.


We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full.

Here are what the columns represent:
* credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
* purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
* int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11).
    Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
* installment: The monthly installments owed by the borrower if the loan is funded.
* log.annual.inc: The natural log of the self-reported annual income of the borrower.
* dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
* fico: The FICO credit score of the borrower.
* days.with.cr.line: The number of days the borrower has had a credit line.
* revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
* revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
* inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
* delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
* pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# --- STEP 1: DATA ANALYSIS
pd.set_option("display.max_columns", None)

df = pd.read_csv('./loan_data.csv')

print(df.head())

"""
As we can notice from the histogram generated below, 
those who have a high fico value respect, at the same moment,
the website financial policies. 
So the tree may understand that these users likely give back money. 
"""
df[df['credit.policy']==1]['fico'].hist(bins=30, alpha=0.5, color='red', label='Credit Policy 1')
histplot_1 = df[df['credit.policy']==0]['fico'].hist(bins=30, alpha=0.5, color='blue', label='Credit Policy 1')
plt.legend()
plt.show()

"""
This plot confirms what said above: 
those who have a positive credit policy, 
almost always give back money
"""
df[df['credit.policy']==1]['not.fully.paid'].hist(bins=30, alpha=0.5, color='red', label='Credit Policy 1')
histplot_2 = df[df['credit.policy']==0]['not.fully.paid'].hist(bins=30, alpha=0.5, color='blue', label='Credit Policy 0')
plt.legend()
plt.show()

countplot = sns.countplot(x='purpose',hue='not.fully.paid',data=df,palette='Set1')
plt.show()

# --- STEP 2: TRAIN/TEST SPLIT AND PREDICTION WITH A SINGLE TREE

df_final = pd.get_dummies(df, columns=['purpose'], drop_first=True)

X = df_final.drop('not.fully.paid',axis=1)
y = df['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)

prediction = tree.predict(X_test)

print(classification_report(y_test, prediction))
"""
Output:
              precision    recall  f1-score   support

           0       0.84      0.82      0.83      1607
           1       0.17      0.19      0.18       309

    accuracy                           0.72      1916
   macro avg       0.51      0.51      0.51      1916
weighted avg       0.73      0.72      0.73      1916

The trained model got a decent result on predicting who gives back money. 
On 1607 considered cases, the tree successfully predicted the 82% of them. So it generated only 
18% of false positives: that means it considered users as bad payers even they weren't. 

On the other hand, the tree had critical performance on prediciting who might be a bad payer. 
It generated an 81% of false negatives. We can conclude, without any doubts, this is caused by the unbalanced dataset
"""

# --- STEP 3: FOREST
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=200)
forest.fit(X_train, y_train)
prediction = forest.predict(X_test)

print(classification_report(y_test, prediction))
"""
Output:
              precision    recall  f1-score   support

           0       0.83      0.99      0.91      1593
           1       0.53      0.03      0.05       323

    accuracy                           0.83      1916
   macro avg       0.68      0.51      0.48      1916
weighted avg       0.78      0.83      0.76      1916

The forest can be considered infallible in predicting who gives money back, 
but had unacceptable performances on the opposite side. On 323 bad payers, the forest
considered the 97% as good payer. 
"""
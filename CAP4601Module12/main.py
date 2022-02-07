# pip install sklearn
# pip install pydotplus
# pip install Graphviz
# pip install pandas
# pip install xlrd

# pip install IPython

# pip install matplotlib


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

file = r"/Users/jerry/PycharmProjects/CAP4601Module11/venv/steak-risk-survey.xls"
df = pd.read_excel(file)

print("name of columns", df.columns)


# print(df)


###########################
# 1


# Action

def drink(x):
    if x == 'Yes':
        return 1
    else:
        return 0


df['drinker'] = df['Do you ever drink alcohol?'].map(lambda x: drink(x))

ActionDf = df['drinker'].map({0: "No", 1: "Yes"}).astype(str)

print(ActionDf)


def age_range(x):
    if x == '18-29':
        return 1
    elif x == '30-44':
        return 2
    elif x == '45-60':
        return 3
    elif x == '> 60':
        return 4
    else:
        return 5


df['AgeRange'] = df['Age'].map(lambda x: age_range(x))


def steak_cooked(x):
    if x == 'Rare':
        return 1
    elif x == 'Medium rare':
        return 2
    elif x == 'Medium':
        return 3
    elif x == 'Medium Well':
        return 4
    elif x == 'Well':
        return 5
    else:
        return 6


df['SteakCooked'] = df['How do you like your steak prepared?'].map(lambda x: steak_cooked(x))

data = df.drop(columns=['RespondentID',
                        'Consider the following hypothetical situations: <br>In Lottery A, you have a 50%'
                        ' chance of success, with a payout of $100. <br>In Lottery B, you have a 90% chance '
                        'of success, with a payout of $20. <br><br>Assuming you have $10 to bet, would you play '
                        'Lottery A or Lottery B?',
                        'Do you ever smoke cigarettes?', 'Do you ever drink alcohol?',
                        'Do you ever gamble?', 'Have you ever been skydiving?',
                        'Do you ever drive above the speed limit?',
                        'Have you ever cheated on your significant other?', 'Do you eat steak?',
                        'How do you like your steak prepared?', 'Gender', 'Age',
                        'Household Income', 'Education', 'Location (Census Region)'])
print(data)

# train test split.
X_train, X_test, Y_train, Y_test = train_test_split(data, ActionDf, test_size=0.65, stratify=ActionDf, random_state=1)

# Display Training data
Y_traindf = Y_train.to_frame()
# print(Y_traindf)

Train = pd.concat([X_train, Y_traindf], axis=1, join='inner')
# print(Train)

TrainSmoke = Train.query('drinker=="Yes"')
TrainDontSmoke = Train.query('drinker=="No"')

ax = TrainSmoke.plot.scatter(x='AgeRange', y='SteakCooked', c='green')
TrainDontSmoke.plot.scatter(x='AgeRange', y='SteakCooked', c='red', ax=ax)
plt.title('Training set age range vs steak cooked range')
plt.xlabel('age range')
plt.ylabel('steak cooked range')

plt.show()

# Display Testing data
Y_testdf = Y_test.to_frame()
# print(Y_traindf)
Test = pd.concat([X_test, Y_testdf], axis=1, join='inner')
# print(Train)
TestSmoke = Test.query('drinker=="Yes"')
TestDontSmoke = Test.query('drinker=="No"')

ax = TestSmoke.plot.scatter(x='AgeRange', y='SteakCooked', c='green')
TestDontSmoke.plot.scatter(x='AgeRange', y='SteakCooked', c='red', ax=ax)
plt.title('Testing set age range vs steak cooked range')
plt.xlabel('age range')
plt.ylabel('steak cooked range')

plt.show()

#2

data = df.drop(columns=['RespondentID',
                        'Consider the following hypothetical situations: <br>In Lottery A, you have a 50%'
                        ' chance of success, with a payout of $100. <br>In Lottery B, you have a 90% chance '
                        'of success, with a payout of $20. <br><br>Assuming you have $10 to bet, would you play '
                        'Lottery A or Lottery B?',
                        'Do you ever smoke cigarettes?', 'Do you ever drink alcohol?',
                        'Do you ever gamble?', 'Have you ever been skydiving?',
                        'Do you ever drive above the speed limit?',
                        'Have you ever cheated on your significant other?', 'Do you eat steak?',
                        'How do you like your steak prepared?', 'Gender', 'Age',
                        'Household Income', 'Education', 'Location (Census Region)'])

print(data)

#train test split.
X_train, X_test, Y_train, Y_test = train_test_split(data,ActionDf, test_size = 0.7, stratify = ActionDf, random_state = 1)
#perceptron
print("Perceptron")
perceptron = lm.Perceptron( verbose=1)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)
print("Perceptron")
print("Number of Features...",perceptron.n_features_in_)
print("columns",X_train.columns)
print("coeficients",perceptron.coef_)
print("intercept",perceptron.intercept_)

print('Accuracy of perceptron on test set: {:.2f}'.format(perceptron.score(X_test, Y_test)))

print("confusion_matrix...")
confusion_matrixP = confusion_matrix(Y_test, Y_pred)
print(confusion_matrixP)
###########################
print("###############################")
print("Logistic Regression")
logreg = LogisticRegression(verbose=1)
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)
print("Logistic Regression")
print("columns",X_train.columns)
print("coeficients",logreg.coef_)
print("intercept",logreg.intercept_)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, Y_test)))

print("confusion_matrix...")
confusion_matrixLG = confusion_matrix(Y_test, Y_pred)
print(confusion_matrixLG)

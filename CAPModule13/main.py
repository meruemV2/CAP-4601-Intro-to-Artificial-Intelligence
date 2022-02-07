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
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import csv as csv

file = r"/Users/jerry/PycharmProjects/CAP4601Module11/venv/steak-risk-survey.xls"
df = pd.read_excel(file)

print("name of columns", df.columns)


# print(df)


###########################
# 1


# Action
df = df.dropna()


def drink(x):
    if x == 'Yes':
        return 1
    if x == 'No':
        return 0
    else:
        return 2


df['drinker'] = df['Do you ever drink alcohol?'].map(lambda x: drink(x))

ActionDf = df['drinker'].map({0: 'Drinks', 1: 'Does not drink'}).astype(str)

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
                        'Household Income', 'Education', 'Location (Census Region)', 'drinker'])

print(data)

# 1) simple GaussianNB
# train test split.
X_train, X_test, Y_train, Y_test = train_test_split(data, ActionDf, test_size=0.7, stratify=ActionDf, random_state=1)

model = GaussianNB()
model.fit(X_train, Y_train)

print("Training data")
print("######################## \n")

expected = Y_train
predicted = model.predict(X_train)
print("Naive Bayes")
print("Params", model.get_params())
print("Class labels", model.get_params())

print("Class labels", model.classes_)
print("Probability of each class", model.class_prior_)
print("absolute additive value to variances", model.epsilon_)
print("variance of each feature per class", model.sigma_)
print("mean of each feature per class", model.theta_)

print('Accuracy on test set: {:.2f}'.format(model.score(X_train, Y_train)))
print("confusion_matrix...")
print(metrics.confusion_matrix(expected, predicted))

print("Test data")
print("######################## \n")

expected = Y_test
predicted = model.predict(X_test)
print("Naive Bayes")
print("Params", model.get_params())
print("Class labels", model.get_params())

print("Class labels", model.classes_)
print("Probability of each class", model.class_prior_)
print("absolute additive value to variances", model.epsilon_)
print("variance of each feature per class", model.sigma_)
print("mean of each feature per class", model.theta_)

print('Accuracy on test set: {:.2f}'.format(model.score(X_test, Y_test)))
print("confusion_matrix...")
print(metrics.confusion_matrix(expected, predicted))


# 2A more emphasis on drinkers

# train test split.
print("#################### \n")
X_train, X_test, Y_train, Y_test = train_test_split(data, ActionDf, test_size=0.7, stratify=ActionDf, random_state=1)

model = GaussianNB()

print("#Question 2A more emphasis on drinkers")
sample_weight = Y_train.map({'Drinks': .01, 'Does not drink': .99}).astype(float)

model.fit(X_train, Y_train, sample_weight)

expected = Y_test
predicted = model.predict(X_test)
print("Naive Bayes")
print("Params", model.get_params())

print("Class labels", model.classes_)
print("Probability of each class", model.class_prior_)
print("absolute additive value to variances", model.epsilon_)
print("variance of each feature per class", model.sigma_)
print("mean of each feature per class", model.theta_)
print('Accuracy on test set: {:.2f}'.format(model.score(X_test, Y_test)))
print("confusion_matrix...")
print(metrics.confusion_matrix(expected, predicted))

# 2B more emphasis on non-drinkers


print("\n\nQuestion 2B more emphasis on those who do not drink")
sample_weight = Y_train.map({'Drinks': .99, 'Does not drink': .01}).astype(float)

model.fit(X_train, Y_train, sample_weight)

expected = Y_test
predicted = model.predict(X_test)
print("Naive Bayes")
print("Params", model.get_params())

print("Class labels", model.classes_)
print("Probability of each class", model.class_prior_)
print("absolute additive value to variances", model.epsilon_)
print("variance of each feature per class", model.sigma_)
print("mean of each feature per class", model.theta_)
print('Accuracy on test set: {:.2f}'.format(model.score(X_test, Y_test)))
print("confusion_matrix...")
print(metrics.confusion_matrix(expected, predicted))
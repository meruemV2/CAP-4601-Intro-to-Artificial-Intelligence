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
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import csv as csv

file = "/Users/jerry/Downloads/adult.data"
# above .data file is comma delimited
df = pd.read_csv(file, delimiter=",")
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship',
              'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '>50k or <=50k']

print("number of rows", df.shape[0])
print("number of columns", df.shape[1])
print("name of columns", df.columns)

print(df.info())
df = df.dropna()


def age_range(x):
    if x <= 25:
        return 1
    elif x <= 45:
        return 2
    elif x <= 65:
        return 3
    else:
        return 4


df['AgeRange'] = df['age'].map(lambda x: age_range(x))


def gender(x):
    if x == ' Female':
        return 0
    elif x == ' Male':
        return 1
    else:
        return 2


df['Gender'] = df['sex'].map(lambda x: gender(x))


def education_level(x):
    if x == ' Preschool':
        return 0
    elif x == ' 1st-4th':
        return 0
    elif x == ' 5th-6th':
        return 0
    elif x == ' 7th-8th':
        return 0
    elif x == ' 9th':
        return 1
    elif x == ' 10th':
        return 1
    elif x == ' 11th':
        return 1
    elif x == ' 12th':
        return 1
    elif x == ' HS-grad':
        return 2
    elif x == ' Prof-school':
        return 3
    elif x == ' Assoc-acdm':
        return 3
    elif x == ' Some-college':
        return 3
    elif x == ' Bachelors':
        return 4
    elif x == ' Masters':
        return 5
    elif x == ' Doctorate':
        return 6
    else:
        return 7


df['Education'] = df['education'].map(lambda x: education_level(x))


def hours_worked(x):
    if x <= 25:
        return 1
    elif x <= 40:
        return 2
    elif x <= 60:
        return 3
    else:
        return 4


df['HoursWorked'] = df['hours-per-week'].map(lambda x: hours_worked(x))


def income(x):
    if x == ' <=50K':
        return 0
    elif x == ' >50K':
        return 1
    else:
        return 2


df['income'] = df['>50k or <=50k'].map(lambda x: income(x))
ActionDf = df['income'].map({0: ' <=50K', 1: ' >50K'}).astype(str)

print(ActionDf)

###############################################
data = df.drop(columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country', '>50k or <=50k', 'income'])

X_train, X_test, Y_train, Y_test = train_test_split(data, ActionDf, test_size=0.65, stratify=ActionDf, random_state=1)

# Display Training data
Y_traindf = Y_train.to_frame()
# print(Y_traindf)

Train = pd.concat([X_train, Y_traindf], axis=1, join='inner')
# print(Train)

TrainPoor = Train.query('income==" <=50K"')
TrainRich = Train.query('income==" >50K"')

ax = TrainPoor.plot.scatter(x='AgeRange', y='Education', c='green')
TrainRich.plot.scatter(x='AgeRange', y='Education', c='red', ax=ax)
plt.title('Training set age range vs Education level')
plt.xlabel('age range')
plt.ylabel('education')

plt.show()

Y_testdf = Y_test.to_frame()
# print(Y_traindf)
Test = pd.concat([X_test, Y_testdf], axis=1, join='inner')
# print(Train)
TestPoor = Test.query('income==" <=50K"')
TestRich = Test.query('income==" >50K"')

ax = TestPoor.plot.scatter(x='AgeRange', y='Education', c='green')
TestRich.plot.scatter(x='AgeRange', y='Education', c='red', ax=ax)
plt.title('Testing set age range vs education range')
plt.xlabel('age range')
plt.ylabel('education')

plt.show()

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

################################################
data = df.drop(columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country', '>50k or <=50k', 'income'])

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

#pip install pandas
#pip install wheel
#pip install sklearn
#pip install pydotplus
#pip install xlrd

## import dependencies
import numpy as np
from sklearn import tree #For our Decision Tree
import pandas as pd # For our DataFrame
import pydotplus # To create our Decision Tree Graph
from sklearn.metrics import accuracy_score


'''
name of columns Index(['RespondentID',
       'Consider the following hypothetical situations: <br>In Lottery A, you have a 50% chance of success, with a payout of $100. <br>In Lottery B, you have a 90% chance of success, with a payout of $20. <br><br>Assuming you have $10 to bet, would you play Lottery A or Lottery B?',
       'Do you ever smoke cigarettes?', 'Do you ever drink alcohol?',
       'Do you ever gamble?', 'Have you ever been skydiving?',
       'Do you ever drive above the speed limit?',
       'Have you ever cheated on your significant other?', 'Do you eat steak?',
       'How do you like your steak prepared?', 'Gender', 'Age',
       'Household Income', 'Education', 'Location (Census Region)'],
'''

file = r"/Users/jerry/PycharmProjects/CAP4601Module10/steak-risk-survey.xls"
df = pd.read_excel(file)

def steak_cooked(x):
   if x == 'Rare' or x == 'Medium rare':
       return 'Good'
   elif x == 'Medium' or x == 'Medium Well':
       return 'Ehh'
   else:
       return 'Burger'

df['SteakCooked'] = df['How do you like your steak prepared?'].map(lambda x: steak_cooked(x))


data = pd.get_dummies(df[ ['SteakCooked','Have you ever cheated on your significant other?',
                           'Do you ever drive above the speed limit?' , 'Do you eat steak?' , 'Do you ever drink alcohol?' ] ])

data = data.dropna()

clf = tree.DecisionTreeClassifier(criterion="entropy")

def gender(x):
   if x == 'Male':
       return 'Male'
   else:
       return 'Female'

df['genderBool'] = df['Gender'].map(lambda x: gender(x))


clf_train = clf.fit(data, df['genderBool'])

#Create Dot Data
dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(data.columns.values),
                                class_names=['Male', 'Female'], rounded=True, filled=True) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes
#Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)
######################################################

######################################################
#Example of writing tree to a .pdf file
# Create PDF
graph.write_pdf("DTEST1.pdf")

from info_gain import info_gain

print('Information gain')

ig = info_gain.info_gain(df['genderBool'], df['SteakCooked'])
print('Steak Cooked=', ig)

ig = info_gain.info_gain(df['genderBool'], df['Have you ever cheated on your significant other?'])
print('Have you ever cheated on your significant other?=', ig)

ig = info_gain.info_gain(df['genderBool'], df['Do you ever drive above the speed limit?'])
print('Do you ever drive above the speed limit?=', ig)

ig = info_gain.info_gain(df['genderBool'], df['Do you eat steak?'])
print('Do you eat steak?=', ig)

ig = info_gain.info_gain(df['genderBool'], df['Do you ever drink alcohol?'])
print('Do you ever drink alcohol?=', ig)


data = pd.get_dummies(df[ ['SteakCooked','Have you ever cheated on your significant other?',
                           'Do you ever drive above the speed limit?' , 'Do you eat steak?' , 'Do you ever drink alcohol?' ] ])

# The decision tree classifier.
clf = tree.DecisionTreeClassifier(criterion="entropy")
# Training the Decision Tree
clf_train = clf.fit(data, df['genderBool'])

#Create Dot Data
dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(data.columns.values),
                                class_names=['Male', 'Female'], rounded=True, filled=True) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes
#Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Create PDF
graph.write_pdf("DTEST2.pdf")
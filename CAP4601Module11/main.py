#pip install sklearn
#pip install pydotplus
#pip install Graphviz
#pip install pandas
#pip innstall xlrd

import pandas as pd
import numpy as np
## import dependencies
from sklearn import tree #For our Decision Tree
import pandas as pd # For our DataFrame
import pydotplus # To create our Decision Tree Graph

#pip install IPython
from IPython.display import Image  # To Display a image of our graph

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#pip install matplotlib
import matplotlib.pyplot as plt

file = r"/Users/jerry/PycharmProjects/CAP4601Module11/venv/steak-risk-survey.xls"
df = pd.read_excel(file)
print(df.head())
print("name of columns",df.columns)
print(df.head(10))

def steak_cooked(x):
   if x == 'Rare' or x == 'Medium rare':
       return 'Good'
   elif x == 'Medium' or x == 'Medium Well':
       return 'Ehh'
   else:
       return 'Burger'

df['SteakCooked'] = df['How do you like your steak prepared?'].map(lambda x: steak_cooked(x))

def gender(x):
   if x == 'Male':
       return 'Male'
   else:
       return 'Female'

df['genderBool'] = df['Gender'].map(lambda x: gender(x))


data = pd.get_dummies(df[ ['SteakCooked','Have you ever cheated on your significant other?',
                           'Do you ever drive above the speed limit?' , 'Do you eat steak?' , 'Do you ever drink alcohol?' ] ])


data = data.dropna()

clf = tree.DecisionTreeClassifier(criterion="entropy")


clf_train = clf.fit(data, df['genderBool'])

#Create Dot Data
dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(data.columns.values),
                                class_names=['Male', 'Female'], rounded=True, filled=True) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes
#Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)
######################################################
graph.write_pdf("DTEST1.pdf")


#split data into test and training set
x = data
y = df['genderBool']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

# Training the Decision Tree
clf_train = clf.fit(x_train, y_train)

dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=list(data.columns.values),
                                class_names=['Male', 'Female'], rounded=True, filled=True)

graph = pydotplus.graph_from_dot_data(dot_data)

# Create PDF
graph.write_pdf("DT.pdf")


NumRuns = 6
TrainingSetSize=[]
ScorePer = []
n =0
for per in range(10,55,5):
    TrainingSetSize.append(per*.01)
    ScorePer.append(0)
    for i in range(NumRuns):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=(per*.01),random_state=100)

        clf_train = clf.fit(x_train, y_train)
        pred = clf_train.predict(x_test)
        ScorePer[n] += accuracy_score(y_test, pred)
    ScorePer[n] /=NumRuns
    n+=1

#plot graph
d = pd.DataFrame({
 'accuracy':pd.Series(ScorePer),
 'training set size':pd.Series(TrainingSetSize)})

plt.plot('training set size','accuracy', data=d, label='accuracy')
plt.ylabel('accuracy')
plt.xlabel('training set size')
plt.show()


max_depth = []
entropy = []
for i in range(1,10):
 dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
 dtree.fit(x_train, y_train)
 pred = dtree.predict(x_test)
 entropy.append(accuracy_score(y_test, pred))
 max_depth.append(i)

#plot graph
d = pd.DataFrame({
 'entropy':pd.Series(entropy),
 'max_depth':pd.Series(max_depth)})

plt.plot('max_depth','entropy', data=d, label='entropy')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.legend()
plt.show()
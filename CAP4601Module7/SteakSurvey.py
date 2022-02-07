#!/usr/bin/python3
#pip install pandas
#pip install xlrd
#pip install PrettyTable
import pandas as pd
import numpy as np
#install ptable
from prettytable import PrettyTable

file = r"//Users/jerry/PycharmProjects/CAP4601Module7/venv/steak-risk-survey.xlsx"
df = pd.read_excel(file)

#print(df)
print("number of rows", df.shape[0])
print("number of columns", df.shape[1])
print("name of columns", df.columns)

print(df.info())
print(df['Do you ever smoke cigarettes?'].unique())


dfSmoked = df.query('`Do you ever smoke cigarettes?`==\'Yes\'')
percentSmoked = dfSmoked.shape[0] / df.shape[0]
print("Percent of people who smoked =", round(percentSmoked*100.0, 2), "%")

dfDrink = df.query('`Do you ever drink alcohol?`==\'Yes\'')
percentDrink = dfDrink.shape[0] / df.shape[0]
print("Percent of people who drink =", round(percentDrink*100.0, 2), "%")

dfGamble = df.query('`Do you ever gamble?`==\'Yes\'')
percentGamble = dfGamble.shape[0] / df.shape[0]
print("Percent of people who gamble =", round(percentGamble*100.0, 2), "%")


print(df['Do you ever smoke cigarettes?'], df['Do you ever drink alcohol?'], df['Do you ever gamble?'])

x = PrettyTable()
x.field_names = ["","A1 = Smoke", "A2 = Drink", "A3 = Gameble"]
x.add_row(["#", dfSmoked.shape[0], dfDrink.shape[0], dfGamble.shape[0]])
x.add_row(["Out of", df.shape[0], df.shape[0], df.shape[0]])
print(x.get_string(title="Distribution table with attribute count of people who.."))

x = PrettyTable()
x.field_names = ["","A1 = Smoke", "A2 = Drink", "A3 = Gameble"]
x.add_row(["Percentage", round(percentSmoked*100.0, 2), round(percentDrink*100.0, 2), round(percentGamble*100.0, 2)])
print(x.get_string(title="Distribution table with attribute percentages of people who.."))

x = PrettyTable()
x.field_names = ["Drink","Don't Drink", "Gamble", "Don't Gameble"]
x.add_row(["Smoke", dfSmoked.shape[0], dfDrink.shape[0], dfGamble.shape[0]])
x.add_row(["Smoke", df.shape[0], df.shape[0], df.shape[0]])
print(x.get_string(title="Distribution table with attribute count of people who.."))

dfSmokeDrink = df.query('`Do you ever smoke cigarettes?`==\'Yes\' and `Do you ever drink alcohol?`==\'Yes\'')
dfSmokeDontDrink = df.query('`Do you ever smoke cigarettes?`==\'Yes\' and `Do you ever drink alcohol?`==\'No\'')
dfSmokeGamble = df.query('`Do you ever smoke cigarettes?`==\'Yes\' and `Do you ever gamble?`==\'Yes\'')
dfSmokeDontGamble = df.query('`Do you ever smoke cigarettes?`==\'Yes\' and `Do you ever gamble?`==\'No\'')
dfDontSmokeDrink = df.query('`Do you ever smoke cigarettes?`==\'No\' and `Do you ever drink alcohol?`==\'Yes\'')
dfDontSmokeDontDrink = df.query('`Do you ever smoke cigarettes?`==\'No\' and `Do you ever drink alcohol?`==\'No\'')
dfDontSmokeGamble = df.query('`Do you ever smoke cigarettes?`==\'No\' and `Do you ever gamble?`==\'Yes\'')
dfDontSmokeDontGamble = df.query('`Do you ever smoke cigarettes?`==\'No\' and `Do you ever gamble?`==\'No\'')

dfGambleDrink = df.query('`Do you ever drink alcohol?`==\'Yes\' and `Do you ever gamble?`==\'Yes\'')
dfGambleDontDrink = df.query('`Do you ever drink alcohol?`==\'No\' and `Do you ever gamble?`==\'Yes\'')


dfDrinkTotal = dfSmokeDrink.shape[0] + dfDontSmokeDrink.shape[0]
dfDontDrinkTotal = dfSmokeDontDrink.shape[0] + dfDontSmokeDontDrink.shape[0]
dfGambleTotal = dfSmokeGamble.shape[0] + dfDontSmokeGamble.shape[0]
dfDontGambleTotal = dfSmokeDontGamble.shape[0] + dfDontSmokeDontGamble.shape[0]


dfSmokeTotal = dfSmokeGamble.shape[0] + dfDontSmokeDontGamble.shape[0]
dfDontSmokeTotal = dfDontSmokeGamble.shape[0] + dfDontSmokeDontGamble.shape[0]

x = PrettyTable()
x.field_names = ["","Drink", "Don't Drink", "Gamble", "Don't Gamble"]
x.add_row(["Smoke", dfSmokeDrink.shape[0], dfSmokeDontDrink.shape[0], dfSmokeGamble.shape[0], dfSmokeDontGamble.shape[0]])
x.add_row(["Don't Smoke", dfDontSmokeDrink.shape[0], dfDontSmokeDontDrink.shape[0], dfDontSmokeGamble.shape[0], dfDontSmokeDontGamble.shape[0]])
x.add_row(["Total", dfDrinkTotal, dfDontDrinkTotal, dfGambleTotal, dfDontGambleTotal])
print(x.get_string(title="Distribution Table "))

x = PrettyTable()
x.field_names = ["","Drink", "Don't Drink", "Gamble", "Don't Gamble"]
x.add_row(["Smoke", round(dfSmokeDrink.shape[0]/df.shape[0]*100.0,2), round(dfSmokeDontDrink.shape[0]/df.shape[0]*100.0,2), round(dfSmokeGamble.shape[0]/df.shape[0]*100.0,2), round(dfSmokeDontGamble.shape[0]/df.shape[0]*100.0,2)])
x.add_row(["Don't Smoke", round(dfDontSmokeDrink.shape[0]/df.shape[0]*100.0,2), round(dfDontSmokeDontDrink.shape[0]/df.shape[0]*100.0,2), round(dfDontSmokeGamble.shape[0]/df.shape[0]*100.0,2), round(dfDontSmokeDontGamble.shape[0]/df.shape[0]*100.0,2)])
x.add_row(["Total", round(dfDrinkTotal/df.shape[0]*100.0,2), round(dfDontDrinkTotal/df.shape[0]*100.0,2), round(dfGambleTotal/df.shape[0]*100.0,2), round(dfDontGambleTotal/df.shape[0]*100.0,2)])
print(x.get_string(title="Distribution Table (Percentages)"))

print("P(Smoke)", round((dfSmokeDrink.shape[0] + dfSmokeDontDrink.shape[0])/df.shape[0]*100.0,2))
print("P(Drink)", round(dfDrinkTotal/df.shape[0]*100.0,2))
print("P(Gamble)", round(dfGambleTotal/df.shape[0]*100.0,2))

pSmokeorDrink = round((((dfSmokeDrink.shape[0] + dfSmokeDontDrink.shape[0])/df.shape[0]) + dfDrinkTotal/df.shape[0] - dfSmokeDrink.shape[0]/df.shape[0])*100.0,3)
print("P(Smoke V Drink)", pSmokeorDrink)
print("P(Smoke and Gamble)", round(dfSmokeGamble.shape[0]/df.shape[0]*100.0,2))
dfSDG = df.query('`Do you ever smoke cigarettes?`==\'Yes\' and '
                 '`Do you ever drink alcohol?`==\'Yes\' and `Do you ever gamble?`==\'Yes\'')
pDrinkBarSmokeandGamble = (dfSDG.shape[0]/dfSmokeGamble.shape[0])
print("P(Drink | Smoke and Gamble)", round(pDrinkBarSmokeandGamble*100.0,2))

pBayes = ((pDrinkBarSmokeandGamble * dfSmokeGamble.shape[0])/ dfDrinkTotal)
print("P(Smoke and Gamble | Drink)", round(pBayes*100.0,2))


pDrinkSmoke = round(dfSmokeDrink.shape[0]/df.shape[0]*1.0,2)
pSmoke = round(dfSmoked.shape[0] / df.shape[0]*1.0,2)

pDrinkDontSmoke = round(dfDontSmokeDrink.shape[0]/df.shape[0]*1.0,2)
pDontSmoke = round((df.shape[0] - dfSmoked.shape[0])/df.shape[0]*1.0,2)

pDgivenS = round(pDrinkSmoke/pSmoke*1.0,2)
pDgivenNotS = round(pDrinkDontSmoke/pDontSmoke*1.0,2)

print("P(D | S): ", pDgivenS)
print("P(D | Not S): ", pDgivenNotS)



pGambleDrink = round(dfGambleDrink.shape[0]/df.shape[0]*1.0,2)
pDrink = round(dfDrinkTotal/df.shape[0]*1.0,2)

pGambleDontDrink = round(dfGambleDontDrink.shape[0]/df.shape[0]*1.0,2)
pDontDrink = round((df.shape[0] - dfDrinkTotal)/df.shape[0]*1.0,2)

pGgivenD = round(pGambleDrink/pDrink*1.0,2)
pGgivenNotD = round(pGambleDontDrink/pDontDrink*1.0,2)

print("P(G | D): ", pGgivenD)
print("P( G | Not D): ", pGgivenNotD)


#print("P(D | S): ", round((dfSmokeDrink.shape[0]/df.shape)/(dfSmoked.shape[0]/df.shape) *100.0,2))

#print("P(D | not S): ", round(dfDontSmokeDrink.shape[0]/(df.shape[0]- dfSmoked.shape[0]) *100.0,2))

#print("P(G | D): ", round(dfGambleDrink.shape[0]/dfDrink.shape[0] *100.0,2))

import pandas as pd

dataframe = pd.read_csv("Fraud_check.csv")

fraud = []
for value in dataframe["Taxable.Income"]:

    if value <= 30000:
        fraud.append("risky")
    else:
        fraud.append("good")
dataframe["Fraud"] = fraud
dataframe.drop(['City.Population'], axis=1,inplace=True)
dataframe_v1 =pd.get_dummies(dataframe, columns=["Undergrad","Marital.Status","Urban"])


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X= dataframe_v1.iloc[:,[0,1,3,4,5,6,7,8,9]]
y= dataframe_v1.iloc[:,2]

x2_train, x2_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


model=DecisionTreeClassifier(criterion="entropy",max_depth=100,random_state=1)
model.fit(x2_train,y_train)
y_pred=model.predict(x2_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(accuracy_score(y_test,y_pred))
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(model.fit(x2_train, y_train))
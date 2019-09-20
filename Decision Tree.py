import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

xl=pd.read_excel('test.xlsx')
x=xl.iloc[:,0:7].values
y=xl.iloc[:,8].values

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=.20,random_state=17)
logreg = DecisionTreeClassifier()
assert isinstance(Y_train, object)
logreg.fit(X_train,Y_train)
y_pred=logreg.predict(X_test)
accuracy=accuracy_score(Y_test,y_pred)
print("Test Accuracy: ",accuracy*100)

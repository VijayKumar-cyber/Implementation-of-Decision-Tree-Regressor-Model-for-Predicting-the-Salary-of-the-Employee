# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the dataset (handle missing values, encode categorical variables, split features and target).

2.Split the data into training and testing sets.

3.Train a Decision Tree Regressor on the training data.

4.Predict and evaluate the model on the test data using metrics like MAE or R² score.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: VIJAY KUMAR D
RegisterNumber:25000878  
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
```
```
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
```
```
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
```
```
print("R2 score: ", r2)
```
```
import pandas as pd
dt.predict(pd.DataFrame([[5, 6]], columns=["Position", "Level"]))
```


## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
<img width="250" height="45" alt="Screenshot 2025-10-29 105747" src="https://github.com/user-attachments/assets/1473a100-5e76-4d44-947f-2bcc2ca78a22" />

<img width="385" height="60" alt="Screenshot 2025-10-29 105740" src="https://github.com/user-attachments/assets/b3f0e623-fcb2-49db-ae1d-84abd3f93d90" />

<img width="186" height="310" alt="Screenshot 2025-10-29 105729" src="https://github.com/user-attachments/assets/2544969f-c5e4-4729-be36-e6ad882a35ac" />

<img width="387" height="265" alt="Screenshot 2025-10-29 105721" src="https://github.com/user-attachments/assets/d80b70eb-87a4-4bba-99b0-d66eb301b1cb" />

<img width="439" height="449" alt="Screenshot 2025-10-29 105711" src="https://github.com/user-attachments/assets/f8bbf8a5-6108-4e38-bd06-c1f16ade7fe2" />




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

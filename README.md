# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize weights, bias, learning rate, and epochs.
2. Update weights using SGD for each training sample.
3. Predict outputs using the trained model.
4. Compute error metrics to evaluate performance.

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: POOJA U
RegisterNumber: 212225230209
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print(data.info())

data=data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)
X=data.drop('price',axis=1)
y=data['price']

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1,1))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sgd_model=SGDRegressor(max_iter=1000,tol=1e-3)

sgd_model.fit(X_train,y_train)
y_pred=sgd_model.predict(X_test)

print('Name: POOJA U ')
print('Reg. No: 212225230209')

print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"R^2: {r2_score(y_test,y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.2f}")

print("\nModel Coefficients:")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.grid(True)
plt.show() 
*/
```

## Output:
<img width="1331" height="770" alt="Screenshot 2026-02-25 083134" src="https://github.com/user-attachments/assets/26d71e94-53fa-4b4d-979a-762e0ee33b47" />

<img width="1082" height="758" alt="Screenshot 2026-02-25 083149" src="https://github.com/user-attachments/assets/ccb0eed0-ce5e-4533-867d-74319cefbdd4" />

<img width="894" height="334" alt="Screenshot 2026-02-25 083206" src="https://github.com/user-attachments/assets/bc5266df-b697-4249-b259-fd06e6dfa213" />
<img width="1385" height="680" alt="Screenshot 2026-02-25 082330" src="https://github.com/user-attachments/assets/ddae8147-1827-42da-a581-5f61067ff935" />
<img width="1300" height="781" alt="Screenshot 2026-02-25 083222" src="https://github.com/user-attachments/assets/0ca239d2-32e8-403a-b4a8-c92ea450c175" />


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.

# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and load the dataset by separating input features and target values.
2. Split the dataset into training and testing sets.
3. Create and train the SGD Regressor model using the training data.
4. Predict the output for test data and evaluate the model performance using error metrics. 

## Program:
```
/*
data=pd.get_dummies(data,drop_first=True)
X=data.drop('price',axis=1)
y=data['price']
scaler=StandardScaler()
X=scaler.fit_transform(X)
y=scaler.fit_transform(np.array(y).reshape(-1,1))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
sgd_model=SGDRegressor(max_iter=1000,tol=1e-3)
sgd_model.fit(X_train,y_train)
y_pred=sgd_model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2= r2_score(y_test,y_pred)
print('Name:Sanjay A ')
print('Reg No: 212225040367')
print("Mean Squared Error",mse)
print("R-squared Score:",r2)
print("\nModel Coefficients:")
print("Coefficients:",sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.show()
Program to implement SGD Regressor for linear regression.
Developed by: Sanjay A
RegisterNumber:  212225040367
*/
```

## Output:
![alt text](<Screenshot 2026-02-25 082708.png>)

<img width="712" height="232" alt="exp4" src="https://github.com/user-attachments/assets/f8add86b-3dbf-4816-992f-bb0d5a13118f" />


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.


## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.

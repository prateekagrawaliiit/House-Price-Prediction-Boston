#importing dependencies
import pandas as pd
import numpy as np
from sklearn import linear_model 
from sklearn.model_selection import train_test_split

#loading boston dataset
from sklearn.datasets import load_boston
boston = load_boston()
# print(boston)

df_x=pd.DataFrame(boston['data'],columns=boston['feature_names'])
df_y =pd.DataFrame(boston['target'])
# print(df_x)

# print(df_x.describe())

reg= linear_model.LinearRegression()

X_train,X_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.33,random_state = 42)
reg.fit(X_train,y_train)

prediction = reg.predict(X_test)

# print(y_test)
from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test,prediction))
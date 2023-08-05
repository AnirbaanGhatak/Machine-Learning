#Aim: Implementation of Sumple Regression
#Name: Anirbaan Ghatak
#Roll no.: C026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('USA_Housing.csv')

df.head(10)
df.info()
df.columns

sns.heatmap(df.corr(), annot=True)

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
Y = df['Price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.40, random_state=101)

lm = LinearRegression()
lm.fit(X_train, Y_train)

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)

prediction = lm.predict(X_test)
plt.scatter(Y_test, prediction)

print('SSE:', np.sum((Y_test - prediction) ** 2))
print('RSME:', np.sqrt(mean_squared_error(Y_test, prediction)))
print('R2:', r2_score(Y_test, prediction))
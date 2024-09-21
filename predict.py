import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('canada_per_capita_income.csv')

X = data[['year']]
y = data['per capita income (US$)']

model = LinearRegression()

model.fit(X, y)
print(f"Predict Canada's per capita income in year 2020: {model.predict([[2020]])}")

plt.xlabel('Year')
plt.ylabel('Per Capita Income (US$)')
plt.scatter(X, y, color='red')

plt.show()

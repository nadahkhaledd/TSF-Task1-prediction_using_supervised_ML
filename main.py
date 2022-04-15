
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


dataset = pd.read_csv('data.csv')
x = dataset['Hours']
y = dataset['Scores']

plt.title('Raw Data')
plt.scatter(x, y, color='black')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0)

regression = LinearRegression()
regression.fit(X=xTrain.reshape(-1, 1),  y=yTrain)

line = regression.coef_ * x + regression.intercept_
plt.title('Gradient descent Line fit')
plt.scatter(x, y, color='black')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.plot(x, line, color='red')
plt.show()

predicted = regression.predict(X=xTest)

MSE = metrics.mean_squared_error(yTest, predicted)

print('mean square error: {}'.format(MSE))


newData = 9.25
newTest = np.array([newData])
newTest = newTest.reshape(-1, 1)
newPredicted = regression.predict(X=newTest)
print('predicted score for {} hours is: {}'.format(newData, newPredicted[0]))








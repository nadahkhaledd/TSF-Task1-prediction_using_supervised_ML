
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



dataset = pd.read_csv('data.csv')

x = dataset['Hours']
y = dataset['Scores']

dataset.plot(x='Hours', y='Scores', style='o')
plt.show()


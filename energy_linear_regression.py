import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#read CSV file and assign it to Data variable
data = read_csv('Kenneth_FYP_Benchmark_Data.csv')
#print(data)

# #Extract parameters
parameters = data[['clk_frequency', 'Instruction_Cache_Total_Accessses','Instruction_Cache_Hits_Number', 'Data_Cache_Total_Accessses', 'Data_Cache_Hits_Number', 'Instruction_per_clock_cycle']]
#print(parameters)

# #Extract Energy
energy = data['Energy(mJ)']
#print(energy)

# #Split data into test sets and training sets
parameters_train, parameters_test, energy_train, energy_test = train_test_split(parameters, energy, random_state = 2)
# print(parameters_train)
# print(parameters_test)
# print(energy_train)
# print(energy_test)

#Perform Linear Regression
linear_model = linear_model.LinearRegression()
linear_model.fit(parameters_train, energy_train)

#Use the trained model to do prediction
energy_prediction = linear_model.predict(parameters_test)
print(energy_prediction)
print(energy_test)
print('Model score:', linear_model.score(parameters_test, energy_test))

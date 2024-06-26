import torch
import numpy as np
import pandas as pd
from linear_classification import prepare_data

print(torch.cuda.is_available())

data = pd.read_csv('data/project2_dataset.csv')

# Question 1:

# a) 12330
print(data.shape[0])

# b) 15.47% = 1908 / 12330
print(data[data['Revenue'] == True].shape[0] / data.shape[0] * 100)

# c) 84.52%
print(data[data['Revenue'] == False].shape[0] / data.shape[0] * 100)
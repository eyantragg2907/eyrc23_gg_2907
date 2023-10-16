"""
A test script to load the JITTED model and see if it works
"""

import pandas
import torch
import numpy
from sklearn.model_selection import train_test_split 
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder

modl = torch.jit.load("task_1a_trained_model.pth")

tensin = torch.tensor([[1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 0.0, 1.0, 1.0, 1.0, 2.0, -3.0]])
print(tensin.shape)

ans = modl(tensin)

expected = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
# correct = ((ans.max(1)[1] == labels).sum())

print(ans.max(1)[1])

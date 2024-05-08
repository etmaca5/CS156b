import numpy as np
import sklearn
import torch


a = np.array([1, 2, 3])
print(a)
for i in range(10):
    print(i)
print("finished main")
print(torch.cuda.is_available())

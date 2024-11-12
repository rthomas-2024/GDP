import numpy as np

arr1 = np.array([1,2,3,4,5])

arrZer = np.zeros([len(arr1)])

for i in range(len(arr1)):
    arrZer[i] = 1

print(arrZer)
import numpy as np
import sys

N = int(input())
M = int(input())

arr = np.zeros(N*N-2)
mask = np.random.choice(N*N-2, size=M, replace=False)
arr[mask] = 1
arr = arr.reshape((1,arr.shape[0]))
tmp = np.append(np.zeros((1,1)), arr, axis=1)
tmp = np.append(tmp, np.asarray([2]).reshape((1,1)), axis=1)
matrix = tmp.reshape((N,N))

np.save(sys.argv[1], matrix)

import numpy as np
import matplotlib.pyplot as plt

ALPHA = 2
MU = 0.001
SIG = 1
def f(x, y):
    if x<=0:
        return ALPHA*1.0/(1-x)+y
    elif x>0 and x<ALPHA+y:
        return ALPHA+y
    else:
        return -1

x_n = 500
y_n = -1

X = []
N = []
Y = []
for n in range(2000):
    old_x = x_n
    x_n = f(x_n, y_n)
    y_n = y_n - MU*(old_x+1) + MU*SIG
    X.append(x_n)
    Y.append(y_n)
    N.append(n+1)

fig = plt.figure()
plt.plot(N,X)
plt.xlabel(r'$n$')
plt.ylabel(r'$x_n$')
plt.savefig('i.png', dpi=fig.dpi)

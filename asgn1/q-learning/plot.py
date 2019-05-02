import matplotlib.pyplot as plt
import numpy as np
x1 = np.load('m1perf.npy')
x2 = np.load('m2perf.npy')
x3 = np.load('m3perf.npy')

y = np.arange(500)

fig = plt.figure()
plt.plot(y, x1)
plt.plot(y, x2)
plt.plot(y, x3)
plt.legend([r'$m = 0$', r'$m = 10$', r'$m = 20$' ])
plt.xlabel('Episodes')
plt.ylabel('Steps to reach goal')
plt.title('$n = 5$')

fig.savefig('m.png', dpi=fig.dpi)

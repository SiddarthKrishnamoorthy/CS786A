import numpy as np
import sys

SIZE = 1
print("Enter outputs of boolean function in order")

ip = input()
ip = ip.split(' ')
if ip[-1] == "":
    ip.pop()


'''
# TODO: REMOVE
ip = [1, 1, 0, 1, 0, 0, 1, 0]
'''
ip = [int(x) for x in ip]
ip = np.asarray(ip)
ip = ip.reshape((8,1))

tt = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
tt = np.asarray(tt)

dd = np.append(tt, ip, axis=1)

tmp = np.tile(dd,(SIZE,1))
np.random.shuffle(tmp)
ones = np.ones((SIZE*8, 1))
#tmp = np.append(ones, tmp, axis=1)

def sigmoid(x):
    return 1.0/(1+np.exp(-1*x))

W1 = np.random.normal(0, 0.1, (3,5))
W2 = np.random.normal(0, 0.1, (5,1))
b1 = np.random.normal(0, 0.1, (1,5))
b2 = np.random.normal(0, 0.1, (1,1))

def forward(x, pred=False):
    z1 = np.dot(x, W1)+b1
    x1 = sigmoid(z1)
    z2 = np.dot(x1, W2) + b2
    x2 = sigmoid(z2)

    if pred:
        return x2
    return (z1, x1, z2, x2)

data = tmp[:,:-1]
label = tmp[:,-1]
lr = 0.1
iters = 5000
for j in range(iters):
    print("Epoch number: {0}".format(j))
    sys.stdout.flush()
    dW2 = 0
    dW1 = 0
    db1 = 0
    db2 = 0
    #for i in range(tmp.shape[0]):
    z1, x1, z2, x2 = forward(data)
    loss = 0.5*(x2-label)**2

    label = label.reshape((8,1))
    dL = x2 - label
    dz2 = dL*x2*(1-x2)
    dW2 = np.dot(x1.T, dz2)
    db2 = np.dot(dz2.T, np.ones(8).reshape((8,1)))

    #exit()
    dz1 = np.dot(dz2, W2.T)*x1*(1-x1)
    dW1 = np.dot(data.T, dz1)
    db1 = np.dot(np.ones(8).reshape((1,8)), dz1)

    print("Current loss value: {0}".format(loss))
    sys.stdout.flush()

    W1 = W1 - lr*dW1
    W2 = W2 - lr*dW2
    b2 = b2 - lr*db2
    b1 = b1 - lr*db1

print(forward(data[0].reshape((1,3)), pred=True), label[0])
print(forward(data[1].reshape((1,3)), pred=True), label[1])
print(forward(data[2].reshape((1,3)), pred=True), label[2])
print(forward(data[3].reshape((1,3)), pred=True), label[3])
print(forward(data[4].reshape((1,3)), pred=True), label[4])
print(forward(data[5].reshape((1,3)), pred=True), label[5])
print(forward(data[6].reshape((1,3)), pred=True), label[6])
print(forward(data[7].reshape((1,3)), pred=True), label[7])

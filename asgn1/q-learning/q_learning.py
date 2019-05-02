import numpy as np
import sys
import matplotlib.pyplot as plt

SOFT = False
mat = np.load(sys.argv[2])
SIZE = mat.shape
dim = SIZE[0]*SIZE[1]
N = SIZE[0]
Q = np.zeros(SIZE+(4,))

def possible_moves(curr, mat):
    move_list = []
    if curr[1]<N-1:
        move_list.append(((curr[0], curr[1]+1),0))
    else:
        move_list.append((curr,-1))
    if curr[0]<N-1:
        move_list.append(((curr[0]+1, curr[1]),1))
    else:
        move_list.append((curr,-1))
    if curr[1]>0:
        move_list.append(((curr[0], curr[1]-1),2))
    else:
        move_list.append((curr,-1))
    if curr[0]>0:
        move_list.append(((curr[0]-1, curr[1]),3))
    else:
        move_list.append((curr,-1))

    return move_list

def reward(pos, mat):
    if pos[0]<0 or pos[0]>N-1 or pos[1]<0 or pos[1]>N-1:
        return 0
    if mat[pos[0],pos[1]] == 0:
        return 0
    elif mat[pos[0],pos[1]] == 1:
        return -100
    elif mat[pos[0],pos[1]] == 2:
        return 100

alpha = 0.1
lamda = 0.5

epsilon = 1

curr_pos = (0,0)
move_list = [0,1,2,3]

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

X = []
y = []
perf = []

for ep_count in range(500):
    ctr = 0
    curr_pos = (0,0)
    total_reward = 0
    for i in range(6000):
        if curr_pos == (N-1,N-1):
            break
        tmp = []
        l = possible_moves(curr_pos, mat)

        [pos, moves] = list(zip(*l))
        moves = list(moves)

        '''
        invalid_moves = list(set(move_list) - set(moves))
        for x in invalid_moves:
            Q[curr_pos[0],curr_pos[1],x] = -np.inf
        '''

        # e-greedy
        if not SOFT:
            tmp = np.random.uniform(0, 1)
            if tmp < epsilon:
                a = np.random.randint(4)
            else:
                a = np.argmax(Q[curr_pos[0], curr_pos[1], :])
                #print(Q[curr_pos[0], curr_pos[1], :].shape)

            n = int(a)
        # Softmax
        else:
            a = np.random.multinomial(1,softmax(Q[curr_pos[0], curr_pos[1], :]),size=1)
            sys.stdout.flush()
            a = a.astype(np.bool)

            n = np.arange(4)[a[0]]
            n = int(n)

        next_pos = pos[n]

        R = reward(next_pos, mat)
        total_reward += R

        Q[curr_pos[0],curr_pos[1],n] = (1-alpha)*Q[curr_pos[0],curr_pos[1],n] + alpha*(R+lamda*max(Q[next_pos[0],next_pos[1],:]))
        curr_pos = next_pos
        ctr+=1
        epsilon = 1.0/(ep_count+1)

    X.append(ep_count)
    y.append(total_reward)
    perf.append(ctr)

#print(X)
print(y)
#print("=======================")
#print(Q)
'''
fig = plt.figure()
plt.plot(X, y)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Reward vs Episodes")
fig.savefig('reward.png', dpi=fig.dpi)
plt.close()
fig2 = plt.figure()
plt.plot(X, perf)
plt.xlabel("Episodes")
plt.ylabel("Steps to reach goal")
plt.title("Steps takes to find goal vs Episodes")
fig2.savefig('performance.png', dpi=fig2.dpi)
'''

np.save(sys.argv[1], np.asarray(perf))

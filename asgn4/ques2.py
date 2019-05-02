import numpy as np
from drawFromADist import drawFromADist

# the temporal context model assumes that the past becomes increasingly
# dissimilar to the future, so that memories become harder to retrieve the
# farther away in the past they are

N_WORLD_FEATURES = 5
N_ITEMS = 10
ENCODING_TIME = 500
TEST_TIME = 20

# we are going to model the world as a set of N continuous-valued features.
# we will model observations of states of the world as samples from N
# Gaussians with time-varying means and fixed variance. For simplicity,
# assume that agents change nothing in the world.

# setting scheduling load to last 10 time units increases load but improves the retrieval
schedule = np.arange(10)
schedule = schedule + 490
#schedule = np.sort(np.random.randint(0, ENCODING_TIME+1, size=N_ITEMS))
schedule_load = ENCODING_TIME/np.median(np.diff(schedule, 1)) # variable important for parts 2,3 of assignment
encoding = np.zeros((N_ITEMS, N_WORLD_FEATURES+1))

world_m = np.asarray([1, 2, 1, 2, 3]) # can generate randomly for yourself
world_var = 1
#delta = 0.05 # what does this parameter affect?
beta_param = 0.001 # what does this parameter affect?
m = 0 # state of the world ?

# Gaussian mixture model
mu = [0, 1]
sig = 1
mix_prob = 0.5

def sampleFromGMM():
    s = np.random.binomial(1, mix_prob)
    return np.random.normal(mu[s], sig)
#s = np.random.binomial(1, mix_prob)

# simulating encoding

for time in range(ENCODING_TIME):
    delta = sampleFromGMM()
    world_m = world_m + delta
    world = world_var*np.random.randn(world_m.shape[0]) + world_m
    # any item I want to encode in memory, I encode in association with the
    # state of the world at that time.
    if (m<N_ITEMS):
        if (time == schedule[m]):
            # encode into the encoding vector
            encoding[m, :] = np.append(world, m)
            m += 1

out = np.zeros(TEST_TIME)
while (time < ENCODING_TIME + TEST_TIME):
    # the state of the world is the retrieval cue
    world_m = world_m + delta
    world = world_var*np.random.randn(world_m.shape[0]) + world_m

    soa = np.zeros(N_ITEMS)
    for m in range(N_ITEMS):
        # finding association on strengths
        soa[m] = np.dot(encoding[m], np.append(world, m))
    #soa = soa/np.linalg.norm(soa)

    out[time-ENCODING_TIME] = drawFromADist(soa)
    time += 1

success = np.unique(out)
print("Scheduling load = {0}".format(schedule_load))
print("Number of unique retrievals = {0}".format(success.shape[0]))

import numpy as np

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

# first fix the presentation schedule; I'm assuming its random
schedule = np.sort(np.random.randint(0, ENCODING_TIME+1, size=N_ITEMS))
schedule_load = ENCODING_TIME/np.median(np.diff(schedule, 1)) # variable important for parts 2,3 of assignment
encoding = np.zeros((N_ITEMS, N_WORLD_FEATURES+1))

world_m = np.asarray([1, 2, 1, 2, 3]) # can generate randomly for yourself
world_var = 1
delta = 0.05 # what does this parameter affect?
beta_param = 0.001 # what does this parameter affect?
m = 0

# simulating encoding

for time in range(ENCODING_TIME):
    world_m = world_m + delta
    world = world_var*np.random.randn(world_m.shape) + world_m
    # any item I want to encode in memory, I encode in association with the
    # state of the world at that time.
    if (m<N_ITEMS):
        if (time == schedule[m]):
            # encode into the encoding vector
            encoding[m, :] = 
            m += 1

while (time < ENCODING_TIME + TEST_TIME):
    # the state of the world is the retrieval cue

    for m in range(N_ITEMS):
        # finding association on strengths
        soa(m) = 

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import multiprocessing as mp
import os

np.random.seed(1)

# 1 dimensional
mu0 = 5
mu1 = 10
cov = 10000
c = (mu0 + mu1)/2
np.set_printoptions(threshold=np.inf)

"""
data: The x values
datasequence: The y values (0/1) corresponding to each x in data
"""
def get_erms(data, datasequence):
    sum_0s = 0
    num_0s = 0
    sum_1s = 0
    num_1s = 0

    thetahats = np.zeros(len(data))
    n = 0
    for x, y in zip(data, datasequence):
        if y == 0:
            sum_0s += x
            num_0s += 1
        else:
            sum_1s += x
            num_1s += 1

        if num_0s == 0:
            thetahat = 0#sum_1s/num_1s
        elif num_1s == 0:
            thetahat = 0#sum_0s/num_0s
        else:
            thetahat = (sum_1s/num_1s + sum_0s/num_0s)/2
        thetahats[n] = thetahat
        n += 1
    return thetahats



    n = len(datasequence)
    c_ests = np.zeros(n) # The estimated locations of the best separator

    raw_data_head = np.array([]) # each entry is just x, but sorted
    data_head = np.empty(shape=(0, 3)) # each entry is (x, y, loss) sorted by x.

    # Algorithm to efficiently get all n of the ERMS; done in O(n) time total
    n = 0
    xy_pairs = list(zip(data, datasequence))
    for x, y in xy_pairs:# zip(data, datasequence):
        idx = np.searchsorted(raw_data_head, x, side = "left")

        regret = None
        if len(data_head) == 0:
            regret = y
        else:
            if idx == 0: # Our new point is the smallest x
                if y == 0: # Suppose we get a - as the min x.
                    # Our new data point has 1 more regret than its neighbor if the neighbor is a -; otherwise it has 1 less regret than its neighbor if the neighbor is a +
                    if data_head[idx][1] == 0:
                        if data_head[idx][0] == x: # Special case where new point is the same as its neighbor
                            regret = data_head[idx][2]
                        else:
                            regret = data_head[idx][2] + 1
                    else:
                        regret = data_head[idx][2] - 1
                    # Everything else keeps its current regret.

                else: # Suppose we get a + as the min x
                    # The new point has the same regret as its  neighbor if the neighbor is also a +; otherwise, it worsens the regret by 2 if the neighbor is a -.
                    if data_head[idx][1] == 1:
                        if data_head[idx][0] == x:  # Special case where new point is same as its neighbor
                            regret = data_head[idx][2] + 1
                        else:
                            regret = data_head[idx][2]
                    else:
                        regret = data_head[idx][2] + 2
                    # Furthermore, every other point needs to add 1 to its current regret
                    for datum in data_head[idx:]:
                        datum[2] += 1
            elif idx == len(data_head): # Our new point is the largest x
                if y == 0: # If we get a - as the max x, the regret is the same as its neighbor, and everything else adds 1 to its regret
                    regret = data_head[idx - 1][2]
                    for datum in data_head[0:idx]:
                        datum[2] += 1
                else: # If we get a + as the max x, its regret is one more than its neighbor; everything else keeps the same regret
                    regret = data_head[idx - 1][2] + 1
            else: # The new point has two neighbors
                if y == 0: # If the new point is a -, its regret is the same as its neighbor to the left; everything to the left adds 1 to the regret
                    if data_head[idx][0] == x: # Special case: right neighbor is same as new point
                        regret = data_head[idx][2]
                    else:
                        regret = data_head[idx-1][2]
                    for datum in data_head[0:idx]:
                        datum[2] += 1
                else: # If the new point is a +, its regret is always one more than its neighbor to the left; everything to the right adds 1 to the regret
                    if data_head[idx][0] == x: # Special case: right neighbor is same as new point
                        regret = data_head[idx][2] + 1
                    else:
                        regret = data_head[idx-1][2] + 1
                    for datum in data_head[idx:]:
                        datum[2] += 1

        data_head = np.insert(data_head, idx, [x, y, regret], axis = 0)
        raw_data_head = np.insert(raw_data_head, idx, x)

        c_ests[n] = data_head[np.argmin(data_head[:,2])][0]
        n += 1
    return c_ests

def loss(z, theta):
    x, y = z
    return 1 if ((x <= theta and y == 1) or (x > theta and y == 0)) else 0

def emp_risk(theta, data):
    return np.mean([loss(z, theta) for z in data])

def log_gue_over_omega_fn(data, true_value):
    data1s = [(x, 1) for (x, y) in data if y == 1]
    data0s = [(x, 0) for (x, y) in data if y == 0]

    data_train = np.array(data0s[:len(data0s)//2] + data1s[:len(data1s)//2])
    data_test = np.array(data0s[len(data0s)//2:] + data1s[len(data1s)//2:])

    thetahat = (np.mean([x for (x, y) in data0s][:len(data0s)//2]) + np.mean([x for (x, y) in data1s][:len(data1s)//2]))/2
    return -1 * len(data_test) * (emp_risk(thetahat, data_test) - emp_risk(true_value, data_test))

def offline_gue(data, true_value, alpha):
    bootstrap_iters = 100
    omegas = np.linspace(0, 0.8, 10000)

    data1s = [(x, 1) for (x, y) in data if y == 1]
    data0s = [(x, 0) for (x, y) in data if y == 0]

    data_train = np.array(data0s[:len(data0s)//2] + data1s[:len(data1s)//2])
    data_test = np.array(data0s[len(data0s)//2:] + data1s[len(data1s)//2:])

    boot_truth = (np.mean([x for (x, y) in data0s][:len(data0s)//2]) + np.mean([x for (x, y) in data1s][:len(data1s)//2]))/2
    coverages = np.zeros(len(omegas))
    for boot_iter in range(bootstrap_iters):
        while True:
            boot_data = data_train[np.random.choice(len(data_train), size = len(data_train), replace = True)]
            num_0s = 0
            num_1s = 0
            for datum, dist in boot_data:
                if dist == 0:
                    num_0s += 1
                if dist == 1:
                    num_1s += 1
            if num_0s >= 2 and num_1s >= 2:
                break
        lgoo = log_gue_over_omega_fn(boot_data, boot_truth)
        for idx, omega in enumerate(omegas):
            coverages[idx] += (omega * lgoo < np.log(1/alpha))
    coverages /= bootstrap_iters

    omega = omegas[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])]

    #print()
    #print("    ", omega, coverages[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])], alpha)
    #print()
    return omega * log_gue_over_omega_fn(data, true_value) < np.log(1/alpha)

def online_gue(data, true_value, alpha):
    boot_iters = 100
    coverages = []
    omegas = np.linspace(0, 0.8, num = 1000)

    omega_hats = np.zeros(len(data))
    boot_mus = get_erms([x for (x, y) in data], [y for (x, y) in data])

    for n in range(0, len(data)):
        num_0s = 0
        num_1s = 0
        for datum, dist in data[:n+1]:
            if dist == 0:
                num_0s += 1
            if dist == 1:
                num_1s += 1
        if num_0s < 2 or num_1s < 2:
            omega_hats[n] = 0
            continue

        boot_mu = (np.mean([x for (x, y) in data[:n+1] if y == 0]) + np.mean([x for (x, y) in data[:n+1] if y == 1]))/2

        if n >= 100:
            head = [x for x in omega_hats[first_nonzero:100]]
            omega_hats[n] = np.min(head)
            continue

        coverages = np.zeros(len(omegas))
        for _ in range(boot_iters):
            while True:
                boot_data = data[np.random.choice(n+1, n+1, replace = True)]
                num_0s = 0
                num_1s = 0
                for datum, dist in boot_data:
                    if dist == 0:
                        num_0s += 1
                    if dist == 1:
                        num_1s += 1
                if num_0s >= 2 and num_1s >= 2:
                    break


            boot_erms = get_erms([x for (x, y) in boot_data], [y for (x, y) in boot_data])
            boot_erms = [7.5] + boot_erms

            excess_losses = [loss(z, thetahat) - loss(z, boot_mu) for (thetahat, z) in zip(boot_erms, boot_data)]
            log_gue_over_omega = sum([-1*excess_losses[i] for i in range(n-1)])
            for idx, omega in enumerate(omegas):
                log_gue = omega*log_gue_over_omega
                coverages[idx] += log_gue < np.log(1/alpha)
        coverages /= boot_iters
        omega_hats[n] = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]
        #print(omega_hats[n], coverages[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])])

    erms = [7.5] + boot_mus
    excess_losses = [loss(z, thetahat) - loss(z, true_value) for (thetahat, z) in zip(erms, data)]
    #print(len(data), omega_hats)
    return sum([-1*omega_hat * excess_loss for (omega_hat, excess_loss) in zip(omega_hats, excess_losses)]) < np.log(1/alpha)

def mc_iteration(nom_coverage, iteration_num):
    np.random.seed(iteration_num)

    num_tries = 1
    exact_coverage = 0
    while True:
        data = np.array([])
        datasequence = np.array([])
        n0 = 0
        n1 = 0
        for _ in range(100):
            if np.random.rand() < 0.5:
                data = np.append(data, st.norm.rvs(loc = mu0, scale = cov**0.5))
                datasequence = np.append(datasequence, 0)
                n0 += 1
            else:
                data = np.append(data, st.norm.rvs(loc = mu1, scale = cov**0.5))
                datasequence = np.append(datasequence, 1)
                n1  += 1

        if n0 == 0 or n1 == 0:
            continue

        # Point estimate for (mu0 + mu1)/2
        c_est = (np.mean([x for (x, y) in zip(data, datasequence) if y == 0]) + np.mean([x for (x, y) in zip(data, datasequence) if y == 1]))/2

        # If the estimator is large, try again
        if  c_est > -10:
            num_tries += 1
            continue

        # (1-\alpha)100% confidence interval for (mu0 + mu1)/2
        z_alpha = st.norm.interval(nom_coverage)[1]
        CI = [c_est - z_alpha* ((n0 + n1)/(4*n0*n1) * cov)**0.5, c_est + z_alpha * ((n0 + n1)/(4*n0*n1) * cov)**0.5]
        # Otherwise, we stop collecting data and check if the confidence interval contains the true (mu0+mu1)/2
        if CI[0] <= c and c <= CI[1]:
            exact_coverage = 1
        break

    # Universal interval
    print("    Num Tries:", str(num_tries))#, os.getpid())
    online_coverage = 0#online_gue(np.array(list(zip(data, datasequence))), c, 1 - nom_coverage)
    offline_coverage = offline_gue(np.array(list(zip(data, datasequence))), c, 1 - nom_coverage)
    return (exact_coverage, online_coverage, offline_coverage)


nom_coverages = np.linspace(0, 1, num=100)[90:-1]
exact_coverages = []
online_coverages = []
offline_coverages = []

for nom_coverage in nom_coverages:
    break
    print("Nominal coverage:", round(nom_coverage, 2))
    mc_iters = 100#300
    with mp.Pool(4) as p:
        output = p.starmap(mc_iteration, [(nom_coverage, i) for i in range(mc_iters)])

    exact_coverages.append(np.mean([e for (e, on, off) in output]))
    online_coverages.append(np.mean([on for (e, on, off) in output]))
    offline_coverages.append(np.mean([off for (e, on, off) in output]))
    print(exact_coverages, online_coverages, offline_coverages)

nom_coverages = [0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
exact_coverages = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.011, 0.152, 0.28, 0.41, 0.515, 0.654, 0.773, 0.896]
online_coverages = [0.908, 0.909, 0.909, 0.91, 0.909, 0.91, 0.91, 0.911, 0.915, 0.918, 0.918, 0.972, 0.973, 0.973, 0.975, 0.976, 0.997, 0.998, 0.998, 1.0]
offline_coverages = [0.911, 0.911, 0.911, 0.911, 0.911, 0.911, 0.912, 0.912, 0.913, 0.913, 0.915, 0.972, 0.972, 0.972, 0.972, 0.972, 0.993, 0.995, 0.995, 1.0]

plt.rcParams['text.usetex'] = True
plt.title("Effects of Optional Stopping of a True Null Hypothesis")
plt.scatter(nom_coverages, exact_coverages, color="blue", label = "Exact CI")
plt.scatter(nom_coverages, online_coverages, color = "red", marker = "^", label = "Online GUe CS")
plt.scatter(nom_coverages, offline_coverages, color = "gold", marker = "+", label = "Offline GUe CS")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.legend()
plt.savefig("reroll.svg")

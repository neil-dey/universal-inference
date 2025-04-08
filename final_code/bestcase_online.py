import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

np.random.seed(1)

# 1 dimensional
mu0 = 5
mu1 = 10
cov = 10000
c = (mu0 + mu1)/2
np.set_printoptions(threshold=np.inf)

def emp_risk(theta, data):
    x, y = data
    return 1 if ((x <= theta and y == 1) or (x > theta and y == 0)) else 0

"""
mu: The actual mean of the population data
data: The x values
datasequence: The y values (0/1) corresponding to each x in data
"""
def _gibbs(mu, data, datasequence):
    n = len(datasequence)
    c_ests = np.zeros(n + 1) # The estimated locations of the best separator; start at 0

    raw_data_head = np.array([]) # each entry is just x, but sorted
    data_head = np.empty(shape=(0, 3)) # each entry is (x, y, loss) sorted by x.

    # Algorithm to efficiently get all n of the ERMS; done in O(n) time total
    n = 1
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

    return [emp_risk(thetahat, z) - emp_risk(mu, z) for (thetahat, z) in zip(c_ests, xy_pairs)]

def gibbs(mu, data, datasequence, nom_coverage):
    alpha = 1 - nom_coverage
    boot_iters = 100
    coverages = []
    omegas = np.linspace(0.001, 10, num = 10000)

    # Find the minimum sample size we can work with
    num_0s = 0
    num_1s = 0
    min_n = 0
    for datum, dist in zip(data, datasequence):
        min_n += 1
        if dist == 0:
            num_0s += 1
        if dist == 1:
            num_1s += 1
        if num_0s != 0 and num_1s != 0:
            break

    omega_hats = np.zeros(len(data)+1)
    step_size = 1#max([1, len(data)//200])
    for n in range(min_n + 1, len(data), step_size):
        coverages = np.zeros(len(omegas))
        if n < 100:
            for _ in range(boot_iters):
                # Bootstrap the first n data points (as long as an estimate can still be made from it)
                while True:
                    choices = np.random.choice(range(n), size = n, replace = True)
                    boot_data = data[choices]
                    boot_datasequence = datasequence[choices]

                    sum_0s = 0
                    num_0s = 0
                    sum_1s = 0
                    num_1s = 0
                    for datum, dist in zip(boot_data, boot_datasequence):
                        if dist == 0:
                            sum_0s += datum
                            num_0s += 1
                        if dist == 1:
                            sum_1s += datum
                            num_1s += 1
                    if num_0s != 0 and num_1s != 0:
                        break

                # Estimate the seperator
                boot_c = (sum_0s/num_0s + sum_1s/num_1s)/2
                # Get the losses incurred by the bootstrapped data
                excess_losses = _gibbs(boot_c, boot_data, boot_datasequence)
                # Compute the GUe-value on the first n-1 points, sinc the learning rate is already determined
                log_gue_nminus1 = sum([omega_hats[i]*excess_losses[i] for i in range(n-1)])

                for idx, omega in enumerate(omegas):
                    log_gue = log_gue_nminus1 + omega*excess_losses[-1]
                    coverages[idx] += log_gue < np.log(1/alpha)
            coverages /= boot_iters
            best_omega = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]
            for i in range(step_size):
                if n+i < len(omega_hats):
                    omega_hats[n+i] = st.trim_mean(np.append(omega_hats[:n+i], best_omega), 0.0)#*(1-1.01**-(n+1))
                    #omega_hats[n+i] = best_omega#*(1-1.01**-(n+1))
                    #print(n, omega_hats[n+i], (1-1.01**-(n+1)))
        else:
            omega_hats[n] = st.trim_mean(omega_hats[:n], 0.0)
            if n == 100:
                print(omega_hats[:n])
                print(st.trim_mean(omega_hats[:n], 0.0))

    print(omega_hats[:100])
    print(omega_hats[-100:])
    print(np.mean(omega_hats))
    excess_losses = _gibbs(mu, data, datasequence)
    return sum([omega_hat * excess_loss for (omega_hat, excess_loss) in zip(omega_hats, excess_losses)]) < np.log(1/alpha)
    #return _gibbs(mu, data, datasequence, nom_coverage, omega)

def mc_iteration(nom_coverage, iteration_num):
    print(iteration_num)
    # Exact confidence interval
    exact_coverage = 0
    universal_coverage = 0
    data = np.array([])
    datasequence = np.array([])
    n0 = 0
    n1 = 0
    while True:
        # If making a point estimate is possible
        if n0 >= 2 and n1 >= 2:
            # Point estimate for (mu0 + mu1)/2
            c_est = (np.mean([x for (x, y) in zip(data, datasequence) if y == 0]) + np.mean([x for (x, y) in zip(data, datasequence) if y == 1]))/2

            # (1-\alpha)100% confidence interval for (mu0 + mu1)/2
            z_alpha = st.norm.interval(nom_coverage)[1]
            CI = [c_est - z_alpha* ((n0 + n1)/(4*n0*n1) * cov)**0.5, c_est + z_alpha * ((n0 + n1)/(4*n0*n1) * cov)**0.5]

        # If the confidence interval still includes the origin, or we don't have enough data to make any estimate yet, collect more data and try again
        if (n0 < 2 or n1 < 2) or (CI[0] <= 0 and 0 <= CI[1]):
            if np.random.rand() < 0.5:
                data = np.append(data, st.norm.rvs(loc = mu0, scale = cov**0.5, size = 1))
                n0 += 1
                datasequence = np.append(datasequence, 0)
            else:
                data = np.append(data, st.norm.rvs(loc = mu1, scale = cov**0.5, size = 1))
                n1 += 1
                datasequence = np.append(datasequence, 1)
            continue

        # Otherwise, we stop collecting data and check if the confidence interval contains the true (mu0+mu1)/2
        if CI[0] <= c and c <= CI[1]:
            exact_coverage += 1
        break

    # Universal interval
    print("    ", str(n0+n1))
    universal_coverage = gibbs(c, data, datasequence, nom_coverage)
    print("    ", universal_coverage)
    return (exact_coverage, universal_coverage)


nom_coverages = np.linspace(0, 1, num=100)[80:-1]
nom_coverages = [0.98]
exact_coverages = []
universal_coverages = []
for nom_coverage in nom_coverages:
    print("Nominal coverage:", round(nom_coverage, 2))
    mc_iters = 100#300
    output = [mc_iteration(nom_coverage, i) for i in range(mc_iters)]

    exact_coverages.append(np.mean([ec for (ec, uc) in output]))
    universal_coverages.append(np.mean([uc for (ec, uc) in output]))
    print(exact_coverages, universal_coverages)

# Final results
"""
# Online
0.81 0.49 0.91
0.82 0.49 0.83
0.83 0.55 0.89
0.84 0.61 0.82
0.85 0.56 0.88
0.86 0.69 0.74
0.87 0.64 0.83
0.88 0.61 0.78
0.89 0.73 0.74
0.9 0.77 0.74
0.91 0.73 0.78
0.92 0.71 0.78
0.93 0.8 0.71
0.94 0.77 0.8
0.96 0.82 0.67
0.97 0.93 0.7
"""

"""
exact_coverages = [0.6066666666666667, 0.56, 0.5566666666666666, 0.6433333333333333, 0.64, 0.6233333333333333, 0.64, 0.6733333333333333, 0.7166666666666667, 0.75, 0.7433333333333333, 0.78, 0.8133333333333334, 0.8166666666666667, 0.8733333333333333, 0.8733333333333333, 0.8966666666666666, 0.9466666666666667, 1.0]
universal_coverages = [0.9766666666666667, 0.98, 0.9733333333333334, 0.9766666666666667, 0.9866666666666667, 0.983333333333333, 0.9833333333333333, 0.9966666666666667, 0.9933333333333333, 1.0, 0.99, 0.9966666666666667, 0.9966666666666667, 1.0, 1.0, 0.9966666666666667, 1, 1, 1]
offline_coverages= [0.89, 0.9, 0.9166666666666666, 0.91, 0.8933333333333333, 0.91, 0.9266666666666666, 0.9366666666666666, 0.9433333333333334, 0.9533333333333334, 0.94, 0.9733333333333334, 0.9733333333333334, 0.98, 0.9833333333333333, 0.9833333333333333, 0.9933333333333333, 0.99, 1.0]

plt.rcParams['text.usetex'] = True
#plt.title(r"Stopping Rule: 0 $\not\in\bar{X} \pm z_\alpha \sqrt{s^2/n}$")
plt.title("Effects of Optional Stopping of a True Null Hypothesis")
plt.scatter(nom_coverages, exact_coverages, color="blue", label = "Exact CI")
plt.scatter(nom_coverages, universal_coverages, color = "red", marker = "^", label = "Online GUe CS")
plt.scatter(nom_coverages, offline_coverages, color = "gold", marker = "+", label = "Offline GUe CS")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.legend()
plt.savefig("bestcase.png")
"""

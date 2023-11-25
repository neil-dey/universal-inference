import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

np.random.seed(0)

# 1 dimensional
mu0 = 5
mu1 = 10
cov = 10000
c = (mu0 + mu1)/2


"""
mu: The actual mean of the population data
data: The x values
datasequence: The y values (0/1) corresponding to each x in data
"""
def _gibbs(mu, data, datasequence, nom_coverage, omega):

    n = len(datasequence)
    c_ests = np.zeros(n + 1)

    raw_data_head = np.array([]) # each entry is just x, but sorted
    data_head = np.empty(shape=(0, 3)) # each entry is (x, y, loss) sorted by x.

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

        c_ests[n] = data_head[np.argmin(data_head[:,2])][2]
        n += 1

    def emp_risk(theta, data):
        x, y = data
        return 1 if ((x <= theta and y == 1) or (x > theta and y == 0)) else 0
        #return len([x for (x, y) in data if (x <= theta and y == 1) or (x > theta and y == 0)])

    def T(theta, omega):
        return omega * sum([emp_risk(thetahat, z) - emp_risk(theta, z) for (thetahat, z) in zip(c_ests, xy_pairs)])
        #return omega * sum([((x <= theta and y == 1) or (x > theta and y == 0)) for (x, y) in xy_pairs])
        #return omega * sum([c_est_risks[i] - emp_risk(theta, xy_pairs[0:i+1]) for i in range(len(datasequence))])

    # Check that confidence set contains mu
    return T(mu, omega) >= np.log(1 - nom_coverage)

def gibbs(mu, data, datasequence, nom_coverage):
    boot_iters = 100
    coverages = []
    omegas = np.linspace(0, 1, num = 20)[1:]
    omegas = np.append(omegas, np.linspace(1, 10, num = 20))
    #omegas = np.append(omegas, np.linspace(100, 1000, num = 100))
    #omegas = np.append(omegas, np.linspace(1000, 10000000, num = 100))
    for omega in omegas:
        coverage = 0
        for _ in range(boot_iters):
            while True:
                choices = np.random.choice(range(len(data)), size = len(data), replace = True)
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
            boot_c = (sum_0s/num_0s + sum_1s/num_1s)/2
            coverage += _gibbs(boot_c, boot_data, boot_datasequence, nom_coverage, omega)

        coverage /= boot_iters
        coverages.append(coverage)

    omega = omegas[np.argmin([abs(nom_coverage - coverage) for coverage in coverages])]
    print("  omega:", omega)
    return _gibbs(mu, data, datasequence, nom_coverage, omega)

def mc_iteration(nom_coverage):
    # Exact confidence interval
    exact_coverage = 0
    universal_coverage = 0
    data = np.array([])
    datasequence = np.array([])
    n0 = 0
    n1 = 0
    while True:
        if n0 != 0 and n1 != 0:
            # Point estimate for (mu0 + mu1)/2
            c_est = (np.mean([x for (x, y) in zip(data, datasequence) if y == 0]) + np.mean([x for (x, y) in zip(data, datasequence) if y == 1]))/2

            # (1-\alpha)100% confidence interval for (mu0 + mu1)/2
            z_alpha = st.norm.interval(nom_coverage)[1]
            CI = [c_est - z_alpha* ((n0 + n1)/(4*n0*n1) * cov)**0.5, c_est + z_alpha * ((n0 + n1)/(4*n0*n1) * cov)**0.5]

        # If the confidence interval still includes the origin, collect more data and try again
        if (n0 == 0 or n1 == 0) or (CI[0] <= 0 and 0 <= CI[1]):
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
    universal_coverage = gibbs(c, data, datasequence, nom_coverage)
    return (exact_coverage, universal_coverage)


nom_coverages = np.linspace(0, 1, num=100)[80:-1]
exact_coverages = []
universal_coverages = []
"""
for nom_coverage in nom_coverages:
    print("Nominal coverage:", round(nom_coverage, 2))
    mc_iters = 300
    output = [mc_iteration(nom_coverage) for _ in range(mc_iters)]

    exact_coverages.append(np.mean([ec for (ec, uc) in output]))
    universal_coverages.append(np.mean([uc for (ec, uc) in output]))
    print(exact_coverages, universal_coverages)
"""

exact_coverages = [0.6066666666666667, 0.56, 0.5566666666666666, 0.6433333333333333, 0.64, 0.6233333333333333, 0.64, 0.6733333333333333, 0.7166666666666667, 0.75, 0.7433333333333333, 0.78, 0.8133333333333334, 0.8166666666666667, 0.8733333333333333, 0.8733333333333333, 0.8966666666666666, 0.9466666666666667, 1.0]
universal_coverages = [0.9766666666666667, 0.98, 0.9733333333333334, 0.9766666666666667, 0.9866666666666667, 0.983333333333333, 0.9833333333333333, 0.9966666666666667, 0.9933333333333333, 1.0, 0.99, 0.9966666666666667, 0.9966666666666667, 1.0, 1.0, 0.9966666666666667, 1, 1, 1]
offline_coverages= [0.89, 0.9, 0.9166666666666666, 0.91, 0.8933333333333333, 0.91, 0.9266666666666666, 0.9366666666666666, 0.9433333333333334, 0.9533333333333334, 0.94, 0.9733333333333334, 0.9733333333333334, 0.98, 0.9833333333333333, 0.9833333333333333, 0.9933333333333333, 0.99, 1.0]

plt.rcParams['text.usetex'] = True
plt.title(r"Stopping Rule: 0 $\not\in\bar{X} \pm z_\alpha \sqrt{s^2/n}$")
plt.scatter(nom_coverages, exact_coverages, color="blue", label = "Exact CI")
plt.scatter(nom_coverages, universal_coverages, color = "red", marker = "^", label = "Online GUe CS")
plt.scatter(nom_coverages, offline_coverages, color = "gold", marker = "s", label = "Offline GUe CS")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.legend()
plt.savefig("bestcase.png")

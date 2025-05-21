from sklearn.cluster import KMeans
import scipy.stats as st
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import inv
import multiprocessing as mp
import matplotlib.pyplot as plt

np.random.seed(0)

def kmeans(x):
    mu = KMeans(n_clusters = 3, random_state = 0).fit(x).cluster_centers_

    mu1 = mu[np.argmax(mu, axis = 0)[0]]
    mu = np.delete(mu, np.argmax(mu, axis = 0)[0], axis = 0)
    mu2 = mu[np.argmax(mu, axis = 0)[1]]
    mu3 = mu[np.argmin(mu, axis = 0)[1]]

    return mu1, mu2, mu3

def _gibbs(x, true_mu1, true_mu2, true_mu3, nom_coverage, omega):
    (train, test) = np.vsplit(x, 2)

    # KMeans doesn't like duplicate data points; add a tiny amount of noise to training data to rectify this
    for pt in train:
        pt += (np.random.rand(2,) - 0.5)/1000

    def emp_risk_test(mu, data):
        return sum([np.linalg.norm(xi - mu[np.argmin([np.linalg.norm(xi - mui)**2 for mui in mu])])**2 for xi in data])
    mu1, mu2, mu3 = kmeans(train)
    T = omega * (emp_risk_test([mu1, mu2, mu3], test) - emp_risk_test([true_mu1, true_mu2, true_mu3], test))
    return T >= np.log(1 - nom_coverage)

def gibbs(x, nom_coverage):
    mu1, mu2, mu3 = kmeans(x)

    x = np.array(x)

    boot_iters = 100
    coverages = []
    omegas = np.linspace(0, 100, num = 20)[1:]
    for omega in omegas:
        boot_coverages = [_gibbs(np.random.default_rng().choice(x, size = len(x), shuffle = False), mu1, mu2, mu3, nom_coverage, omega) for _ in range(boot_iters)]
        coverage = np.mean(boot_coverages)
        coverages.append(coverage)

    omega = omegas[np.argmin([abs(nom_coverage - coverage) for coverage in coverages])]
    print("    ", omega, coverages[np.argmin([abs(nom_coverage - coverage) for coverage in coverages])])

    true_mu1 = [1, 0]
    true_mu2 = [np.cos(2*np.pi/3), np.sin(2*np.pi/3)]
    true_mu3 = [np.cos(4*np.pi/3), np.sin(4*np.pi/3)]
    return _gibbs(x, true_mu1, true_mu2, true_mu3, nom_coverage, omega)




def mc_iteration(nom_coverage):
    exact_coverage = 0
    universal_coverage = 0

    # Generate cluster data
    x = []
    sample_size = 10
    cov = 0.01
    #probs = [1/3, 1/2]
    probs = [0.96, 0.75]
    for _ in range(sample_size):
        if np.random.rand() < probs[0]:
            mu = [1, 0]
        elif np.random.rand() < probs[1]:
            mu = [np.cos(2*np.pi/3), np.sin(2*np.pi/3)]
        else:
            mu = [np.cos(4*np.pi/3), np.sin(4*np.pi/3)]
        x.append(st.multivariate_normal.rvs(mean = mu, cov = cov))

    # Bootstrapped CS
    bootstrap_iters = 100
    boot_mu1s = []
    boot_mu2s = []
    boot_mu3s = []
    for boot_iter in range(bootstrap_iters):
        boot_x = np.random.default_rng().choice(x, size = len(x), shuffle = False)
        mu1, mu2, mu3 = kmeans(boot_x)
        boot_mu1s.append(mu1)
        boot_mu2s.append(mu2)
        boot_mu3s.append(mu3)
    mu1_cov_boot = np.cov(boot_mu1s, rowvar = False)
    mu2_cov_boot = np.cov(boot_mu2s, rowvar = False)
    mu3_cov_boot = np.cov(boot_mu3s, rowvar = False)


    mu1, mu2, mu3 = kmeans(x)

    # Transform
    true_mu3 = [np.cos(4*np.pi/3), np.sin(4*np.pi/3)]
    transformed_true_mu3 = inv(sqrtm(mu3_cov_boot)) @ (true_mu3 - mu3)
    sq_dist_to_origin = np.linalg.norm(transformed_true_mu3)**2
    pval = 1 - st.chi2.cdf(sq_dist_to_origin, 2)
    if pval > 1 - nom_coverage:
        exact_coverage += 1

    # Gibbs CS
    true_mu1 = [1, 0]
    true_mu2 = [np.cos(2*np.pi/3), np.sin(2*np.pi/3)]
    true_mu3 = [np.cos(4*np.pi/3), np.sin(4*np.pi/3)]
    universal_coverage += _gibbs(np.array(x), true_mu1, true_mu2, true_mu3, nom_coverage, 30)# gibbs(x, nom_coverage)

    return (exact_coverage, universal_coverage)



nom_coverages = np.linspace(0.01, 1, num=100)[80:-1]
exact_coverages = []
universal_coverages = []
for nom_coverage in nom_coverages:
    print("Nominal Coverage:", nom_coverage)
    mc_iters = 1000
    output = list(map(mc_iteration, [nom_coverage for _ in range(mc_iters)]))

    exact_coverages.append(np.mean([ec for (ec, uc) in output]))
    universal_coverages.append(np.mean([uc for (ec, uc) in output]))
    print(exact_coverages, universal_coverages, flush = True)

# Final results
"""
exact_coverages = [0.036, 0.039, 0.057, 0.045, 0.048, 0.045, 0.044, 0.042, 0.037, 0.047, 0.052, 0.048, 0.051, 0.062, 0.056, 0.064, 0.051, 0.068, 0.066]
universal_coverages = [0.836, 0.857, 0.856, 0.867, 0.892, 0.896, 0.896, 0.895, 0.932, 0.918, 0.932, 0.949, 0.962, 0.967, 0.983, 0.986, 0.986, 0.994, 0.999]
"""

plt.scatter(nom_coverages, exact_coverages, color = "blue", label = "Bootstrapped CS")
plt.scatter(nom_coverages, universal_coverages, color = "red", marker = "^", label = "Offline GUe CS")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.title("Coverage of KMeans Centroid(s)")
plt.legend(loc="center right")
#plt.show()
plt.savefig("kmeans_coverage.png")

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

def online_gue(x, true_mu1, true_mu2, true_mu3, nom_coverage, omega, random = False):
    # KMeans doesn't like duplicate data points; add a tiny amount of noise to data to rectify this
    for pt in x:
        pt += (np.random.rand(2,) - 0.5)/1000

    def emp_risk_test(mu, data):
        return sum([np.linalg.norm(xi - mu[np.argmin([np.linalg.norm(xi - mui)**2 for mui in mu])])**2 for xi in data])

    log_gue = 3
    for i in range(4,len(x)):
        mu1, mu2, mu3 = kmeans(x[:i-1])
        if not random:
            lr = omega
        else:
            lr = np.random.rand()*2*omega
        log_gue += lr * (emp_risk_test([mu1, mu2, mu3], x[:i]) - emp_risk_test([true_mu1, true_mu2, true_mu3],x[:i-1]))
    return log_gue >= np.log(1 - nom_coverage)

def _gibbs(x, true_mu1, true_mu2, true_mu3, nom_coverage, omega, random = False):
    (train, test) = np.vsplit(x, 2)

    # KMeans doesn't like duplicate data points; add a tiny amount of noise to training data to rectify this
    for pt in train:
        pt += (np.random.rand(2,) - 0.5)/1000

    def emp_risk_test(mu, data):
        return sum([np.linalg.norm(xi - mu[np.argmin([np.linalg.norm(xi - mui)**2 for mui in mu])])**2 for xi in data])

    mu1, mu2, mu3 = kmeans(train)
    if random:
        lr = np.random.rand() * 2 * omega
    else:
        lr = omega
    T =  lr * (emp_risk_test([mu1, mu2, mu3], test) - emp_risk_test([true_mu1, true_mu2, true_mu3], test))
    return T >= np.log(1 - nom_coverage)

def mc_iteration(nom_coverage):
    exact_coverage = 0
    online_nonrandom_coverage = 0
    offline_nonrandom_coverage = 0
    online_random_coverage = 0
    offline_random_coverage = 0

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

    """
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
    """
    exact_coverage = 0

    # Gibbs CS
    true_mu1 = [1, 0]
    true_mu2 = [np.cos(2*np.pi/3), np.sin(2*np.pi/3)]
    true_mu3 = [np.cos(4*np.pi/3), np.sin(4*np.pi/3)]
    offline_nonrandom_coverage += _gibbs(np.array(x), true_mu1, true_mu2, true_mu3, nom_coverage, 30, random = False)
    online_nonrandom_coverage += online_gue(np.array(x), true_mu1, true_mu2, true_mu3, nom_coverage, 7, random = False)
    offline_random_coverage += _gibbs(np.array(x), true_mu1, true_mu2, true_mu3, nom_coverage, 30, random = True)
    online_random_coverage += online_gue(np.array(x), true_mu1, true_mu2, true_mu3, nom_coverage, 7, random = True)

    return (offline_nonrandom_coverage,online_nonrandom_coverage, offline_random_coverage, online_random_coverage)



nom_coverages = np.linspace(0.01, 1, num=100)[80:-1]
offline_nonrandom_coverages = []
online_nonrandom_coverages = []
offline_random_coverages = []
online_random_coverages = []
for nom_coverage in nom_coverages:
    print("Nominal Coverage:", nom_coverage)
    mc_iters = 1000
    output = list(map(mc_iteration, [nom_coverage for _ in range(mc_iters)]))

    offline_nonrandom_coverages.append(np.mean([w for (w,x,y,z) in output]))
    online_nonrandom_coverages.append(np.mean([x for (w,x,y,z) in output]))
    offline_random_coverages.append(np.mean([y for (w,x,y,z) in output]))
    online_random_coverages.append(np.mean([z for (w,x,y,z) in output]))
    print(offline_nonrandom_coverages, online_nonrandom_coverages, offline_random_coverages, online_random_coverages, flush = True)

bootstrap_coverages = [0.036, 0.039, 0.057, 0.045, 0.048, 0.045, 0.044, 0.042, 0.037, 0.047, 0.052, 0.048, 0.051, 0.062, 0.056, 0.064, 0.051, 0.068, 0.066]
offline_nonrandom_coverages = [np.float64(0.844), np.float64(0.828), np.float64(0.857), np.float64(0.858), np.float64(0.867), np.float64(0.884), np.float64(0.902), np.float64(0.903), np.float64(0.92), np.float64(0.919), np.float64(0.944), np.float64(0.943), np.float64(0.951), np.float64(0.964), np.float64(0.979), np.float64(0.981), np.float64(0.995), np.float64(0.997), np.float64(0.998)]
online_nonrandom_coverages = [np.float64(0.865), np.float64(0.851), np.float64(0.881), np.float64(0.885), np.float64(0.905), np.float64(0.895), np.float64(0.893), np.float64(0.915), np.float64(0.92), np.float64(0.925), np.float64(0.922), np.float64(0.927), np.float64(0.938), np.float64(0.935), np.float64(0.965), np.float64(0.97), np.float64(0.972), np.float64(0.98), np.float64(0.992)]
offline_random_coverages = [np.float64(0.835), np.float64(0.841), np.float64(0.831), np.float64(0.852), np.float64(0.841), np.float64(0.862), np.float64(0.86), np.float64(0.88), np.float64(0.882), np.float64(0.914), np.float64(0.918), np.float64(0.903), np.float64(0.924), np.float64(0.933), np.float64(0.941), np.float64(0.945), np.float64(0.968), np.float64(0.984), np.float64(0.991)]
online_random_coverages = [np.float64(0.846), np.float64(0.839), np.float64(0.873), np.float64(0.86), np.float64(0.883), np.float64(0.873), np.float64(0.886), np.float64(0.893), np.float64(0.904), np.float64(0.905), np.float64(0.894), np.float64(0.902), np.float64(0.924), np.float64(0.921), np.float64(0.95), np.float64(0.95), np.float64(0.949), np.float64(0.964), np.float64(0.991)]
plt.plot(nom_coverages, nom_coverages, linestyle='dashed', color = 'black')
plt.scatter(nom_coverages, offline_nonrandom_coverages, color ='blue', label = "Offline, ω = 30")
plt.scatter(nom_coverages, online_nonrandom_coverages, color = 'red', marker = "+", label = "Online, ω = 7")
plt.scatter(nom_coverages, offline_random_coverages, color ='m', marker = "x", label = "Offline, ω ~ Unif(0, 60)")
plt.scatter(nom_coverages, online_random_coverages, color = 'g', marker = '^', label = "Online, ω ~ Unif(0, 14)")
plt.scatter(nom_coverages, bootstrap_coverages, color = 'k', marker = '1', label = "Bootstrap")
plt.legend()
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.title("Coverage of KMeans Centroid(s)")
plt.legend(loc="center right")
plt.savefig("kmeans_online.png")

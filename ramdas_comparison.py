import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

np.random.seed(1)

def ramdas(mu, data, alpha, c = 1/2):
    mu_hat = lambda t: (sum([X_i for (i, X_i) in enumerate(data) if i <= t]) + 1/2)/(t+2)
    sigma_sq_hat = lambda t: (sum([(X_i - mu_hat(i))**2 for (i, X_i) in enumerate(data) if i <= t]) + 1/4)/(t+2)
    lam = lambda t: min(c, (2*np.log(2/alpha)/(sigma_sq_hat(t - 1) * (t+1) * np.log(t+2)))**0.5)
    v = lambda t: 4 * (data[t] - mu_hat(t-1))**2
    psi = lambda t: (-np.log(1-t) - t)/4

    t = len(data)
    center = sum([lam(i) * X_i for (i, X_i) in enumerate(data)])/sum([lam(i) for i in range(t)])

    width = (np.log(2/alpha)  + sum([v(i) * psi(lam(i)) for i in range(t)]))/sum([lam(i) for i in range(t)])

    return center - width <= mu and mu <= center+width

def _gibbs(mu, data, alpha, omega):
    data_train = data[:len(data)//2]
    data_test = data[len(data)//2:]
    def risk(theta, data):
        return sum([(theta - X)**2 for X in data])/len(data)

    ratio =  -omega * len(data_test) * (risk(np.mean(data_train), data_test) - risk(mu, data_test))
    return ratio < np.log(1/alpha)

def gibbs(mu, data, alpha):
    boot_iters = 100
    coverages = []
    num_omegas = 100
    omegas = np.linspace(0, 1, num = 100)[1:]
    omegas = np.append(omegas, np.linspace(1, 100, num = 100))
    omegas = np.append(omegas, np.linspace(100, 1000, num = 100))
    omegas = np.append(omegas, np.linspace(1000, 10000000, num = 100))
    for omega in omegas:
        coverage = 0
        for _ in range(boot_iters):
            coverage += _gibbs(mu, np.random.choice(data, size = len(data), replace = True), alpha, omega)
        coverage /= boot_iters
        coverages.append(coverage)

    omega = omegas[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])]
    #print(omega, coverages[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])])
    return _gibbs(mu, data, alpha, omega)



P = st.beta(a=5, b=2)
mu = P.stats()[0]

ramdas_coverages = []
gibbs_coverages = []
nom_coverages = np.linspace(0, 1, num = 100)[1:-1]
"""
for nom_coverage in nom_coverages:
    print(nom_coverage)
    ramdas_coverage = 0
    gibbs_coverage = 0
    mc_iters = 100
    for it in range(mc_iters):
        data = P.rvs(size = 10)
        ramdas_coverage += ramdas(mu, data, 1 - nom_coverage)
        gibbs_coverage += gibbs(mu, data, 1 - nom_coverage)
        print("    ", gibbs_coverage/(it + 1))
    ramdas_coverage /= mc_iters
    gibbs_coverage /= mc_iters
    ramdas_coverages.append(ramdas_coverage)
    gibbs_coverages.append(gibbs_coverage)
    print(ramdas_coverages)
    print(gibbs_coverages)
"""

ramdas_coverages = [0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
gibbs_coverages = [0.58, 0.66, 0.68, 0.65, 0.64, 0.61, 0.7, 0.76, 0.64, 0.64, 0.71, 0.62, 0.66, 0.66, 0.63, 0.56, 0.57, 0.64, 0.69, 0.63, 0.75, 0.58, 0.72, 0.7, 0.69, 0.63, 0.65, 0.71, 0.66, 0.66, 0.63, 0.7, 0.7, 0.68, 0.63, 0.73, 0.66, 0.74, 0.69, 0.64, 0.61, 0.7, 0.77, 0.59, 0.61, 0.65, 0.67, 0.66, 0.65, 0.69, 0.65, 0.7, 0.66, 0.69, 0.7, 0.74, 0.83, 0.79, 0.77, 0.77, 0.78, 0.81, 0.81, 0.88, 0.91, 0.94, 0.92, 0.94, 0.92, 0.98, 0.98, 0.97, 0.98, 0.99, 0.97, 1.0, 0.99, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

plt.scatter(nom_coverages, ramdas_coverages, color = "blue", label = "PrPl-EB")
plt.scatter(nom_coverages, gibbs_coverages, color = "red", label = "Gibbs")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.legend()
plt.show()

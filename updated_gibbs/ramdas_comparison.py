import numpy as np
import scipy.stats as st
from scipy.optimize import brentq
from scipy.integrate import dblquad
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(0)

modes = ["online", "offline"]
mode = modes[0]

omega_modes = ["parambootstrap_normal", "parambootstrap_beta"]
omega_mode = omega_modes[0]

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
    if mode == "online":
        thetahats = [np.mean(data[0: i+1]) for i in range(len(data))]
        thetahats = [0.5] + thetahats
    elif mode == "offline":
        thetahats = [np.mean(data[0:len(data)-1]) for i in range(len(data))]
    else:
        print("Unsupported mode")
        exit()
    ratio =  -omega * sum([(thetahat - x)**2 - (mu - x)**2 for thetahat, x in zip(thetahats, data)])
    return ratio < np.log(1/alpha)

def gibbs(mu, data, alpha):
    boot_iters = 10
    coverages = []
    omegas = np.linspace(0, 100, num = 100)[1:]
    boot_a, boot_b, _, _ = st.fit(st.beta, data, bounds = [(0, 10), (0, 10)]).params
    for omega in omegas:
        coverage = 0
        for _ in range(boot_iters):
            if omega_mode == "parambootstrap_normal":
                boot_data = st.norm(loc = np.mean(data), scale = np.var(data, ddof=1)**0.5).rvs(size = len(data))
            else:
                boot_data = st.beta(a = boot_a, b = boot_b).rvs(size = len(data))
            coverage += _gibbs(np.mean(data), boot_data, alpha, omega)
        coverage /= boot_iters
        coverages.append(coverage)

    omega = omegas[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])]
    #print(omega)

    return _gibbs(mu, data, alpha, omega)



P = st.beta(a=5, b=2)
mu = P.stats()[0]

ramdas_coverages = []
gibbs_coverages = []
nom_coverages = np.linspace(0, 1, num = 100)[80:-1]
for nom_coverage in nom_coverages:
    break
    print(nom_coverage)
    ramdas_coverage = 0
    gibbs_coverage = 0
    mc_iters = 100
    for it in range(mc_iters):
        data = P.rvs(size = 10)
        #ramdas_coverage += ramdas(mu, data, 1 - nom_coverage)
        gibbs_coverage += gibbs(mu, data, 1-nom_coverage)
        #print("    ", gibbs_coverage/(it + 1))
    ramdas_coverage /= mc_iters
    gibbs_coverage /= mc_iters
    ramdas_coverages.append(ramdas_coverage)
    gibbs_coverages.append(gibbs_coverage)
    print(ramdas_coverages)
    print(gibbs_coverages)

"""
offline_halfeq1 = [0.897, 0.906, 0.912, 0.918, 0.938, 0.923, 0.915, 0.938, 0.935, 0.941, 0.955, 0.956, 0.958, 0.968, 0.969, 0.964, 0.986, 0.983, 0.991]
offline_eq1 = [0.788, 0.8, 0.818, 0.804, 0.825, 0.831, 0.822, 0.852, 0.854, 0.87, 0.859, 0.882, 0.883, 0.907, 0.92, 0.924, 0.935, 0.946, 0.961]
online_eq1 = [0.975, 0.981, 0.985, 0.983, 0.988, 0.98, 0.977, 0.979, 0.988, 0.984, 0.987, 0.988, 0.988, 0.987, 0.992, 0.987, 0.995, 0.996, 0.994]
"""
ramdas_coverages = [0.998, 0.996, 1.0, 0.995, 0.998, 0.999, 0.996, 0.998, 1.0, 0.999, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0][-19:]
beta_coverages = [0.79, 0.82, 0.83, 0.83, 0.84, 0.85, 0.85, 0.85, 0.87, 0.89, 0.92, 0.91, 0.91, 0.95, 0.95, 0.95, 0.97, 0.99, 1.0]
normal_coverages = [0.8, 0.8, 0.81, 0.83, 0.84, 0.83, 0.87, 0.87, 0.87, 0.89, 0.89, 0.9, 0.93, 0.92, 0.91, 0.93, 0.96, 0.98, 0.99]

plt.scatter(nom_coverages, ramdas_coverages, color = "blue", label = "PrPl-EB")
plt.scatter(nom_coverages, beta_coverages, color = "red", marker = "^", label = "Online GUe (Beta)")
plt.scatter(nom_coverages, normal_coverages, color = "gold", marker = "s", label = "Online GUe (Normal)")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.title("Coverage of i.i.d. Beta(5, 2) Sample: Online GUe")
plt.legend()
#plt.show()
plt.savefig("online_ramdas.png")

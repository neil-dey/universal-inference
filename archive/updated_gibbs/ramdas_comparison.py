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
mode = modes[1]

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

    return _gibbs(mu, data, alpha, omega)



P = st.beta(a=5, b=2)
mu = P.stats()[0]

ramdas_coverages = []
gibbs_coverages = []
nom_coverages = np.linspace(0, 1, num = 100)[80:-1]
for nom_coverage in nom_coverages:
    print(nom_coverage)
    ramdas_coverage = 0
    gibbs_coverage = 0
    mc_iters = 100
    for it in range(mc_iters):
        data = P.rvs(size = 10)
        ramdas_coverage += ramdas(mu, data, 1 - nom_coverage)
        gibbs_coverage += gibbs(mu, data, 1-nom_coverage)
    ramdas_coverage /= mc_iters
    gibbs_coverage /= mc_iters
    ramdas_coverages.append(ramdas_coverage)
    gibbs_coverages.append(gibbs_coverage)
    print(ramdas_coverages)
    print(gibbs_coverages)


# Final results
"""
ramdas_coverages = [0.998, 0.996, 1.0, 0.995, 0.998, 0.999, 0.996, 0.998, 1.0, 0.999, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0][-19:]
beta_coverages = [0.79, 0.82, 0.83, 0.83, 0.84, 0.85, 0.85, 0.85, 0.87, 0.89, 0.92, 0.91, 0.91, 0.95, 0.95, 0.95, 0.97, 0.99, 1.0]
normal_coverages = [0.8, 0.8, 0.81, 0.83, 0.84, 0.83, 0.87, 0.87, 0.87, 0.89, 0.89, 0.9, 0.93, 0.92, 0.91, 0.93, 0.96, 0.98, 0.99]

plt.scatter(nom_coverages, ramdas_coverages, color = "blue", label = "PrPl-EB")
plt.scatter(nom_coverages, beta_coverages, color = "red", marker = "^", label = "Online GUe (Beta)")
plt.scatter(nom_coverages, normal_coverages, color = "gold", marker = "+", label = "Online GUe (Normal)")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.title("Comparison of PrPl-EB and Online Gue")
plt.legend()
#plt.show()
plt.savefig("online_ramdas.png")
"""

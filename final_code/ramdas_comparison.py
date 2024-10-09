import numpy as np
import scipy.stats as st
from scipy.optimize import brentq
from scipy.integrate import dblquad
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys

np.random.seed(0)

modes = ["online", "offline"]
mode = modes[int(sys.argv[1])]

omega_modes = ["parambootstrap_normal", "parambootstrap_beta", "nonparametric"]
omega_mode = omega_modes[int(sys.argv[2])]

print(mode, omega_mode)

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
        ratio =  -omega * sum([(thetahat - x)**2 - (mu - x)**2 for thetahat, x in zip(thetahats, data)])
    elif mode == "offline":
        thetahats = [np.mean(data[0:len(data)//2]) for i in range(len(data)//2)]
        ratio =  -omega * sum([(thetahat - x)**2 - (mu - x)**2 for thetahat, x in zip(thetahats, data[len(data)//2:])])
    else:
        print("Unsupported mode")
        exit()
    return ratio < np.log(1/alpha)

def gibbs(mu, data, alpha):
    boot_iters = 100
    coverages = []
    omegas = np.linspace(0, 100, num = 100)[1:]

    # For some reason, the online versions use omega = infty a lot to even come close to matching the correct coverage levels on the bootstrapped sapmles at the lower confidence values.
    if mode == "online":
        omegas = np.linspace(0, 100, num = 100)
        omegas[0] = np.inf
    boot_a, boot_b, _, _ = st.fit(st.beta, data, bounds = [(0, 10), (0, 10)]).params
    for omega in omegas:
        coverage = 0
        for _ in range(boot_iters):
            if omega_mode == "parambootstrap_normal":
                boot_data = st.norm(loc = np.mean(data), scale = np.var(data, ddof=1)**0.5).rvs(size = len(data))
            elif omega_mode == "parambootstrap_beta":
                boot_data = st.beta(a = boot_a, b = boot_b).rvs(size = len(data))
            elif omega_mode == "nonparametric":
                n = len(data)
                boot_data = data[np.random.choice(n, n, replace = True)]
            else:
                print("Unsupported Mode")
                exit()
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
    print("Ramdas coverage:", ramdas_coverages)
    print("GUe coverage:", gibbs_coverages)


exit()
# Final Results
ramdas_coverages = [0.998, 0.996, 1.0, 0.995, 0.998, 0.999, 0.996, 0.998, 1.0, 0.999, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0][-19:]
online_beta_coverages = [0.967, 0.967, 0.967, 0.968, 0.968, 0.969, 0.97, 0.967, 0.969, 0.969, 0.97, 0.972, 0.972, 0.972, 0.975, 0.979, 0.98, 0.982, 0.988]
online_normal_coverages = [0.975, 0.974, 0.974, 0.975, 0.977, 0.977, 0.977, 0.978, 0.978, 0.977, 0.978, 0.98, 0.979, 0.981, 0.982, 0.986, 0.988, 0.992, 0.993]
online_nonparametric = [0.979, 0.965, 0.958, 0.963, 0.97, 0.973, 0.96, 0.972, 0.973, 0.969, 0.973, 0.973, 0.97, 0.984, 0.969, 0.977, 0.982, 0.982, 0.993]
plt.scatter(nom_coverages, ramdas_coverages, color = "purple", marker = "x", label = "PrPl-EB")
plt.scatter(nom_coverages, online_beta_coverages, color = "red", marker = "^", label = "Online GUe (Beta)")
plt.scatter(nom_coverages, online_normal_coverages, color = "gold", marker = "+", label = "Online GUe (Normal)")
plt.scatter(nom_coverages, online_nonparametric, color = "blue", label = "Online GUe (Nonparametric)")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.title("Comparison of PrPl-EB and Online Gue")
plt.legend()
plt.savefig("ramdas_online.svg")
plt.clf()

offline_normal_coverages = [0.874, 0.88, 0.889, 0.891, 0.891, 0.897, 0.906, 0.912, 0.919, 0.925, 0.927, 0.932, 0.944, 0.954, 0.959, 0.967, 0.98, 0.988, 0.997]
offline_beta_coverages= [0.872, 0.875, 0.877, 0.88, 0.885, 0.897, 0.902, 0.908, 0.914, 0.918, 0.921, 0.93, 0.935, 0.94, 0.949, 0.958, 0.965, 0.979, 0.993]
offline_nonparametric = [0.857, 0.871, 0.885, 0.884, 0.879, 0.909, 0.893, 0.909, 0.912, 0.93, 0.923, 0.919, 0.936, 0.952, 0.961, 0.952, 0.964, 0.981, 0.994]
plt.scatter(nom_coverages, ramdas_coverages, color = "purple", marker = "x", label = "PrPl-EB")
plt.scatter(nom_coverages, offline_beta_coverages, color = "red", marker = "^", label = "Offline GUe (Beta)")
plt.scatter(nom_coverages, offline_normal_coverages, color = "gold", marker = "+", label = "Offline GUe (Normal)")
plt.scatter(nom_coverages, offline_nonparametric, color = "blue", label = "Offline GUe (Nonparametric)")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.title("Comparison of PrPl-EB and Offline Gue")
plt.legend()
plt.savefig("ramdas_offline.svg")

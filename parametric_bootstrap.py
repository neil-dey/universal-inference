import numpy as np
import scipy.stats as st
from scipy.integrate import dblquad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

np.random.seed(1)


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
    omegas = np.linspace(0, 200, num = 100)[1:]
    """
    omegas = np.append(omegas, np.linspace(1, 100, num = 100))
    omegas = np.append(omegas, np.linspace(100, 1000, num = 100))
    omegas = np.append(omegas, np.linspace(1000, 10000000, num = 100))
    """
    boot_a, boot_b, _, _ = st.fit(st.beta, data, bounds = [(0, 10), (0, 10)]).params
    for omega in omegas:
        coverage = 0
        for _ in range(boot_iters):
            boot_data = st.norm(loc = np.mean(data), scale = np.var(data, ddof=1)**0.5).rvs(size = len(data))#st.beta(a = boot_a, b = boot_b).rvs(size = len(data))
            coverage += _gibbs(np.mean(data), boot_data, alpha, omega)
        coverage /= boot_iters
        coverages.append(coverage)

    #omega = omegas[len(omegas) - np.argmin([abs(1 - alpha - coverage) for coverage in coverages[::-1]]) - 1]
    omega = omegas[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])]
    #print(omega, coverages[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])], boot_a/(boot_a + boot_b))
    return _gibbs(mu, data, alpha, omega)



P = st.beta(a=5, b=2)
mu = P.stats()[0]

gibbs_coverages = []
nom_coverages = np.linspace(0, 1, num = 100)[80:-1]
for nom_coverage in nom_coverages:
    break
    print(nom_coverage)
    gibbs_coverage = 0
    mc_iters = 1000
    for it in range(mc_iters):
        data = P.rvs(size = 10)
        gibbs_coverage += gibbs(mu, data, 1-nom_coverage)#_gibbs(mu, data, 1 - nom_coverage, 410-360*nom_coverage)
        #gibbs_coverage += _gibbs(mu, data, 1 - nom_coverage, 410)
        print("    ", gibbs_coverage/(it + 1))
    gibbs_coverage /= mc_iters
    gibbs_coverages.append(gibbs_coverage)
    print(gibbs_coverages)

ramdas_coverages = [0.998, 0.996, 1.0, 0.995, 0.998, 0.999, 0.996, 0.998, 1.0, 0.999, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0][-19:]
normal_coverages = [0.835, 0.831, 0.836, 0.877, 0.866, 0.857, 0.881, 0.877, 0.879, 0.902, 0.895, 0.908, 0.931, 0.95, 0.942, 0.948, 0.964, 0.966, 0.989] # Assume normal
beta_coverages = [0.82, 0.825, 0.835, 0.854, 0.851, 0.865, 0.852, 0.876, 0.885, 0.894, 0.904, 0.931, 0.902, 0.939, 0.934, 0.954, 0.967, 0.981, 0.989] # Assume beta
plt.scatter(nom_coverages, ramdas_coverages, color = "blue", label = "PrPl-EB")
plt.scatter(nom_coverages, beta_coverages, color = "red", marker = "^", label = "Offline GUe (Beta)")
plt.scatter(nom_coverages, normal_coverages, color = "gold", marker = "s", label = "Offline GUe (Normal)")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.title("Coverage of i.i.d. Beta(5, 2) Sample: Offline Gue")
plt.legend()
plt.savefig("offline_ramdas.png")

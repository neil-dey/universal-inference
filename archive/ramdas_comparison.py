import numpy as np
import scipy.stats as st
from scipy.integrate import dblquad
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import minimize
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


def b(z, omega, alpha, sigma):
    return np.log(1/alpha)/(2*omega*sigma**2 * z) + z/2

def l(z, omega, alpha, sigma, rho, n):
    c_B = 0.4748
    if z == 0:
        return max([0, 1 - c_B*rho/(sigma**3 * n**0.5)])
    return max([0, st.norm.cdf(b(z, omega, alpha, sigma)) - c_B*rho/(sigma**3 * n**0.5)])
def u(z, omega, alpha, sigma, rho, n):
    c_B = 0.4748
    if z == 0:
        return min([1, c_B*rho/(sigma**3 * n**0.5)])
    return min([1, st.norm.cdf(b(z, omega, alpha, sigma)) + c_B * rho/(sigma**3 * n**0.5)])

def safe_choice_lhs(omega, alpha, sigma, rho, n):
    c_B = 0.4748
    h = 1e-6 # step size for numeric derivative

    def l_sp(z):
        return l(z, omega, alpha, sigma, rho, n)
    def u_sp(z):
        return u(z, omega, alpha, sigma, rho, n)
    beta = minimize(l_sp, x0 = [1], bounds = ((1e-6, np.inf),)).x[0]
    gamma = minimize(lambda z: -1*u_sp(z), x0=[-1], bounds = ((-np.inf, -1e-6),)).x[0]
    #print(beta, gamma)

    integral = quad(lambda z: l_sp(z) * (l_sp(z+h) - l_sp(z-h))/(2*h), 1e-6, beta)[0]
    integral += quad(lambda z: u_sp(z) * (l_sp(z+h) - l_sp(z-h))/(2*h), beta, np.inf)[0]
    integral += quad(lambda z: u_sp(z) * (u_sp(z+h) - u_sp(z-h))/(2*h), -np.inf, gamma)[0]
    integral += quad(lambda z: l_sp(z) * (u_sp(z+h) - u_sp(z-h))/(2*h), gamma, -1e-6)[0]

    be = c_B * rho/(sigma**3 * n**0.5)
    lhs = 1 - max([0, 1/2 - be]) - max([0, 1 - be]) + (max([0, 1- be]) + min([1, be])) * min([1, 1/2 + be])

    return lhs + integral


def lr(omega, alpha, sigma, n):
    return 1/(2*np.pi) * dblquad(lambda z1, z2: np.exp(-(z1**2 + z2**2)/2), 0, np.inf, lambda z: b(z, omega, alpha, sigma), np.inf)[0] +  1/(2*np.pi) * dblquad(lambda z1, z2: np.exp(-(z1**2 + z2**2)/2), -np.inf, 0, -np.inf, lambda z: b(z, omega, alpha, sigma))[0]




def _gibbs(mu, data, alpha, omega):
    data_train = data[:len(data)//2]
    data_test = data[len(data)//2:]
    def risk(theta, data):
        return sum([(theta - X)**2 for X in data])/len(data)

    ratio =  -omega * len(data_test) * (risk(np.mean(data_train), data_test) - risk(mu, data_test))
    return ratio < np.log(1/alpha)

def gibbs(mu, data, alpha):
    """
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
            coverage += _gibbs(np.mean(data), np.random.choice(data, size = len(data), replace = True), alpha, omega)
        coverage /= boot_iters
        coverages.append(coverage)

    omega = omegas[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])]
    #print(omega, coverages[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])])
    """
    #sigma = np.var(data, ddof = 1)**0.5
    #omega = fsolve(lambda omega, alpha, sigma: lr(omega, alpha, sigma) - alpha, 1, args = (alpha, sigma))[0]
    sigma = np.mean([x**2 for x in data])**0.5
    rho = np.mean([abs(x**3) for x in data])
    n = len(data)
    #omega = fsolve(lambda omega, alpha, sigma, rho, n: safe_choice_lhs(omega, alpha, sigma, rho, n) - alpha, 1, args = (alpha, sigma, rho, n))[0]
    for omega in np.linspace(0.001, 1000, num = 20):
        print(safe_choice_lhs(omega, alpha, sigma, rho, n))
    exit()
    return _gibbs(mu, data, alpha, omega)



P = st.beta(a=5, b=2)
mu = P.stats()[0]

ramdas_coverages = []
gibbs_coverages = []
nom_coverages = np.linspace(0, 1, num = 100)[80:-1]
for nom_coverage in nom_coverages:
    continue
    print(nom_coverage)
    ramdas_coverage = 0
    gibbs_coverage = 0
    mc_iters = 100
    for it in range(mc_iters):
        data = P.rvs(size = 10)
        ramdas_coverage += ramdas(mu, data, 1 - nom_coverage)
        gibbs_coverage += gibbs(mu, data, 1-nom_coverage)#_gibbs(mu, data, 1 - nom_coverage, 410-360*nom_coverage)
        print("    ", gibbs_coverage/(it + 1))
    ramdas_coverage /= mc_iters
    gibbs_coverage /= mc_iters
    #ramdas_coverages.append(ramdas_coverage)
    gibbs_coverages.append(gibbs_coverage)
    print(ramdas_coverages)
    print(gibbs_coverages)

#ramdas_coverages = [0.998, 0.996, 1.0, 0.995, 0.998, 0.999, 0.996, 0.998, 1.0, 0.999, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0][-19:]


#gibbs_misspec_coverages = [0.835, 0.831, 0.836, 0.877, 0.866, 0.857, 0.881, 0.877, 0.879, 0.902, 0.895, 0.908, 0.931, 0.95, 0.942, 0.948, 0.964, 0.966, 0.989] # Assume normal
#gibbs_spec_coverages = [0.82, 0.825, 0.835, 0.854, 0.851, 0.865, 0.852, 0.876, 0.885, 0.894, 0.904, 0.931, 0.902, 0.939, 0.934, 0.954, 0.967, 0.981, 0.989] # Assume beta

gibbs_eq1_coverages = [0.819, 0.803, 0.819, 0.81, 0.821, 0.841, 0.848, 0.869, 0.853, 0.882, 0.886, 0.888, 0.895, 0.905, 0.924, 0.928, 0.943, 0.949, 0.977]
gibbs_eq1_over2_coverages = [0.875, 0.863, 0.886, 0.872, 0.897, 0.91, 0.908, 0.919, 0.924, 0.937, 0.935, 0.948, 0.965, 0.965, 0.968, 0.968, 0.977, 0.985, 0.993]

#plt.scatter(nom_coverages, ramdas_coverages, color = "blue", label = "PrPl-EB")
#plt.scatter(nom_coverages, gibbs_spec_coverages, color = "red", marker = "^", label = "Offline GUe (Beta)")
#plt.scatter(nom_coverages, gibbs_misspec_coverages, color = "gold", marker = "+", label = "Offline Gue (Normal)")
plt.scatter(nom_coverages, gibbs_eq1_coverages, color = "blue")#, label = "Eq. (4) result")
plt.scatter(nom_coverages, gibbs_eq1_over2_coverages, color = "red", marker = "^")#, label = "Half Eq. (4) result")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.title("Coverage of i.i.d. Beta(5, 2) Sample")
plt.title("Coverage of Closed-Form Learning Rates")
#plt.title("Comparison of PrPl-EB and Offline GUe")
#plt.legend()
#plt.show()
plt.savefig("ramdas_eq1.png")
#plt.savefig("offline_ramdas.png")

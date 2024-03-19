import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import fsolve
import scipy.stats as st
import matplotlib.pyplot as plt

def u(z, omega, alpha, sigma):
    return np.log(1/alpha)/(2*omega*sigma**2 * z) + z/2

def lr(omega, alpha, sigma = 1):
    return 1/(2*np.pi) * dblquad(lambda z1, z2: np.exp(-(z1**2 + z2**2)/2), 0, np.inf, lambda z: u(z, omega, alpha, sigma), np.inf)[0] +  1/(2*np.pi) * dblquad(lambda z1, z2: np.exp(-(z1**2 + z2**2)/2), -np.inf, 0, -np.inf, lambda z: u(z, omega, alpha, sigma))[0]


prob = 0
mc_iters = 1000
alpha = 0.1
n = 50
P = st.norm()
omega = 36
for mc_iter in range(mc_iters):
    data1 = st.multivariate_normal.rvs(mean = [1, 0, np.cos(2*np.pi/3), np.sin(2*np.pi/3), np.cos(4*np.pi/3), np.sin(4*np.pi/3)], cov = .01, size = n)
    data2 = st.multivariate_normal.rvs(mean = [1, 0, np.cos(2*np.pi/3), np.sin(2*np.pi/3), np.cos(4*np.pi/3), np.sin(4*np.pi/3)], cov = .01, size = n)
    thetahat = sum([x for x in data1])/n
    print(thetahat)
    print()
    logG = -omega*sum([np.linalg.norm(Xi - thetahat)**2 - np.linalg.norm(Xi)**2 for Xi in data2])
    if logG >= np.log(1/alpha):
        prob += 1
prob /= mc_iters
print(prob)
exit()

"""
P = st.norm()
n = 100
alpha = 0.8
omega = 30
prob = 0
mc_iters = 10000
for _ in range(mc_iters):
    data1 = P.rvs(size = n)
    data2 = P.rvs(size = n)
    xbar = np.mean(data1)
    thetahat = np.mean(data2)
    if xbar*thetahat - thetahat**2/2 >= np.log(1/alpha)/(2*omega*n):
        prob += 1
prob /= mc_iters
print(prob)
print(lr(omega, alpha))
exit()
"""

P = st.beta(a=5, b=2)
mu = P.stats()[0]
sigma = P.stats()[1]**0.5


nom_coverages = np.linspace(0.01, 1, num = 100)[80:-1]
rates = []
for nom_coverage in nom_coverages:
    alpha = 1 - nom_coverage
    res = fsolve(lambda omega, alpha, sigma: lr(omega, alpha, sigma) - alpha, 1, args = (alpha, sigma))
    rates.append(res[0])

plt.scatter(nom_coverages, rates, color = "blue")
plt.scatter(nom_coverages, [410 - 360*nom_coverage for nom_coverage in nom_coverages], color = "red")
plt.show()



def _gibbs(mu, data, alpha, omega):
    data_train = data[:len(data)//2]
    data_test = data[len(data)//2:]
    def risk(theta, data):
        return sum([(theta - X)**2 for X in data])/len(data)

    ratio =  -omega * len(data_test) * (risk(np.mean(data_train), data_test) - risk(mu, data_test))
    return ratio < np.log(1/alpha)


gibbs_coverages = []
for nom_coverage, rate in zip(nom_coverages, rates):
    print(nom_coverage)
    gibbs_coverage = 0
    mc_iters = 1000
    for it in range(mc_iters):
        data = P.rvs(size = 10)
        gibbs_coverage += _gibbs(mu, data, 1 - nom_coverage, rate)
    gibbs_coverage /= mc_iters
    gibbs_coverages.append(gibbs_coverage)
    print(gibbs_coverages)

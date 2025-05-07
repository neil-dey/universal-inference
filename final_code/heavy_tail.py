import numpy as np
import scipy.stats as st
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import sys
import time
import multiprocessing as mp

np.random.seed(0)

theta = 0
n = 10
mc_iters = 1000

def catoni(true_theta, data, true_variance, alpha):
    def phi(x):
        if x >= 0:
            return np.log(1 + x + x**2/2)
        return -np.log(1 - x + x**2/2)
    def eta_sq(t, s2, alpha):
        return 2*s2 *np.log(2/alpha)/(t - 2*np.log(2/alpha))
    def lambda_sq(t, s2, alpha):
        l2 = 2*np.log(2/alpha)/(t * (s2 + eta_sq(t, s2, alpha)))
        if l2 < 0:
            return min([1/t, 0.1**2])
        return l2

    s2 = true_variance

    lb = -1*s2*sum([lambda_sq(i+1, s2, alpha) for i in range(len(data))])/2 - np.log(2/alpha)
    ub = s2*sum([lambda_sq(i+1, s2, alpha) for i in range(len(data))])/2 + np.log(2/alpha)

    mid = sum([phi(np.sqrt(lambda_sq(i+1, s2, alpha)) * (x - true_theta)) for (i, x) in enumerate(data)])

    return lb <= mid and mid <= ub

def loss(x, theta):
    return (x-theta)**2

def emp_risk(theta, xs):
    return np.mean([loss(x, theta) for x in xs])

def log_gue_over_omega_fn(data, true_value):
    data_train = data[:len(data)//2]
    data_test = data[len(data)//2:]
    return -1 * len(data_test) * (emp_risk(np.mean(data_train), data_test) - emp_risk(true_value, data_test))

def offline_gue(data, true_value, alpha):
    bootstrap_iters = 100
    omegas = np.linspace(0, 1, 100)[1:]
    coverages = np.zeros(len(omegas))

    data_train = data[:len(data)//2]
    data_test = data[len(data)//2:]

    for boot_iter in range(bootstrap_iters):
        boot_data = data_train[np.random.choice(len(data_train), size = len(data_train), replace = True)]
        lgoo = log_gue_over_omega_fn(boot_data, np.mean(data_train))
        for idx, omega in enumerate(omegas):
            coverages[idx] += (omega * lgoo < np.log(1/alpha))
    coverages /= bootstrap_iters

    omega = omegas[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])]

    #print(coverages)
    #print("    ", omega, coverages[np.argmin([abs(1 - alpha - coverage) for coverage in coverages])])
    return omega * log_gue_over_omega_fn(data, true_value) < np.log(1/alpha)

def online_gue(data, true_value, alpha):
    boot_iters = 100
    coverages = []
    omegas = np.linspace(0, 1, num = 100)[1:]

    omega_hats = np.zeros(len(data))
    for n in range(0, len(data)):
        coverages = np.zeros(len(omegas))
        for _ in range(boot_iters):
            boot_data = data[np.random.choice(n+1, n+1, replace = True)]


            boot_erms = [np.mean(boot_data[0: i+1]) for i in range(len(boot_data))]
            boot_erms = [1] + boot_erms
            boot_mu = np.mean(data[:n+1])

            excess_losses = [(thetahat - x)**2 - (boot_mu - x)**2 for (thetahat, x) in zip(boot_erms, boot_data)]
            log_gue_over_omega = sum([-1*excess_losses[i] for i in range(n-1)])
            for idx, omega in enumerate(omegas):
                log_gue = omega*log_gue_over_omega
                coverages[idx] += log_gue < np.log(1/alpha)
        coverages /= boot_iters
        omega_hats[n] = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]

    erms = [np.mean(data[0: i+1]) for i in range(len(data))]
    erms = [1] + erms
    excess_losses = [(thetahat - x)**2 - (true_value - x)**2 for (thetahat, x) in zip(erms, data)]
    return sum([-1*omega_hat * excess_loss for (omega_hat, excess_loss) in zip(omega_hats, excess_losses)]) < np.log(1/alpha)

def gibbs(true_value, data, alpha, mode):
    if mode == "on":
        return online_gue(data, true_value, alpha)
    else:
        return offline_gue(data, true_value, alpha)

def mc_iteration(mc_iter, nom_coverage):
    exact_coverage = 0
    online_gue_coverage = 0
    offline_gue_coverage = 0
    catoni_coverage = 0

    xs = st.t.rvs(df=3, size = n)
    thetahat = np.mean(xs)
    s2 = st.t.stats(df=3)[1]
    catoni_coverage += catoni(theta, xs, s2,  1-nom_coverage/100)

    distances = []
    boot_iters = 100
    for _ in range(boot_iters):
        indices = np.random.choice(n, n)
        boot_xs = xs[indices]
        boot_thetahat = np.mean(boot_xs)
        distances.append(abs(boot_thetahat - thetahat))

    if abs(thetahat - theta) < np.percentile(distances, nom_coverage):
        exact_coverage += 1

    if gibbs(theta, xs, 1-nom_coverage/100, "on"):
        online_gue_coverage += 1

    if gibbs(theta, xs, 1-nom_coverage/100, "off"):
        offline_gue_coverage += 1

    return (exact_coverage, online_gue_coverage, offline_gue_coverage, catoni_coverage)


nom_coverages = np.linspace(0, 1, num=100)[80:-1] * 100
exact_coverages = []
online_gue_coverages = []
offline_gue_coverages = []
catoni_coverages = []
for nom_coverage in nom_coverages:
    with mp.Pool(4) as p:
        coverages = p.starmap(mc_iteration, [(mc_iter, nom_coverage) for mc_iter in range(1000)])
    exact_coverages.append(np.mean([e for (e, on, off, c) in coverages]))
    online_gue_coverages.append(np.mean([on for (e, on, off, c) in coverages]))
    offline_gue_coverages.append(np.mean([off for (e, on, off, c) in coverages]))
    catoni_coverages.append(np.mean([c for (e, on, off, c) in coverages]))

    print(nom_coverage)
    print(exact_coverages)
    print(online_gue_coverages)
    print(offline_gue_coverages)
    print(catoni_coverages)

# Final Results
nom_coverages /= 100
"""
nom_coverages = [0.94, 0.91, 0.98, 0.96, 0.93, 0.95, 0.97, 0.92, 0.89, 0.9, 0.83, 0.82, 0.8, 0.81, 0.87, 0.86, 0.85, 0.88, 0.84]
exact_coverages = [0.897, 0.859, 0.944, 0.921, 0.883, 0.908, 0.931, 0.871, 0.836, 0.848, 0.763, 0.755, 0.729, 0.739, 0.811, 0.795, 0.784, 0.821, 0.772]
online_gue_coverages = [0.975, 0.966, 0.991, 0.981, 0.973, 0.978, 0.985, 0.968, 0.963, 0.967, 0.954, 0.952, 0.951, 0.953, 0.961, 0.96, 0.958, 0.962, 0.957]
offline_gue_coverages = [0.964, 0.937, 0.996, 0.982, 0.952, 0.972, 0.99, 0.944, 0.922, 0.929, 0.892, 0.885, 0.879, 0.882, 0.917, 0.904, 0.902, 0.916, 0.898]
catoni_coverages = [1.0, 0.999, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 1.0, 0.998, 0.996, 0.999, 0.997, 0.992, 0.998, 0.998, 0.999, 0.997, 0.993]
"""

plt.title("Coverage of Mean with Heavy Tails")
plt.scatter(nom_coverages, exact_coverages, color = "blue", label = "Bootstrapped CS")
plt.scatter(nom_coverages, online_gue_coverages, color = "red", marker = "^", label = "Online GUe CS")
plt.scatter(nom_coverages, offline_gue_coverages, color = "gold", marker = "+", label = "Offline GUe CS")
plt.scatter(nom_coverages, catoni_coverages, color = "purple", marker = "x", label = "Catoni CS")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.legend()
plt.savefig("heavy_tail_gue.svg")
plt.show()

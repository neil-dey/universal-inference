import numpy as np
import scipy.stats as st
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import sys
import time
from itertools import compress

np.random.seed(0)


n = 10
boot_iters = 100
mc_iters = 10

theta = 10
sigma_alpha = 1/n
sigma_beta = 1/(2*n)
sigma = 1

nom_coverage = int(sys.argv[1])

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
    omegas = np.linspace(0, 3, 100)[1:]
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
    omegas = np.linspace(0, 3, num = 100)[1:]

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


bootstrap_intervals = []
offline_intervals = []
online_intervals = []
for mc_iter in range(mc_iters):
    candidate_thetas = np.linspace(5.5, 15.5, num=100)
    bootstrap_inclusions = []
    online_inclusions = []
    offline_inclusions = []

    xs = np.array([theta + st.gamma.rvs(a = 1, loc = -sigma_alpha, scale = sigma_alpha) + st.gamma.rvs(a = 1, loc = -sigma_beta, scale = sigma_beta) + st.gamma.rvs(a = 1, loc = -sigma, scale = sigma) for _ in range(n)])
    thetahat = np.mean(xs)

    for c_t in candidate_thetas:
        distances = []
        for _ in range(boot_iters):
            indices = np.random.choice(n, n)
            boot_xs = xs[indices]
            boot_thetahat = np.mean(boot_xs)
            distances.append(abs(boot_thetahat - thetahat))

        bootstrap_inclusions.append(abs(thetahat - c_t) < np.percentile(distances, nom_coverage))

        online_inclusions.append(gibbs(c_t, xs, 1-nom_coverage/100, "on"))

        offline_inclusions.append(gibbs(c_t, xs, 1-nom_coverage/100, "off"))


    l = list(compress(candidate_thetas, bootstrap_inclusions))
    bootstrap_intervals.append((min(l), max(l)))

    l = list(compress(candidate_thetas, online_inclusions))
    online_intervals.append((min(l), max(l)))

    l = list(compress(candidate_thetas, offline_inclusions))
    offline_intervals.append((min(l), max(l)))


print(bootstrap_intervals)
print(online_intervals)
print(offline_intervals)

plt.title("Visualizations of Confidence Intervals for Two-Way Mean")
x = 0.8
legend_flag = True
for (bi, oni, ofi) in zip(bootstrap_intervals, online_intervals, offline_intervals):
    plt.scatter([x, x], [bi[0], bi[1]], color = 'blue', marker = "o", label = "Bootstrapped CI" if legend_flag else "")
    plt.vlines(x, bi[0], bi[1], color = 'blue')
    x += 0.2
    plt.scatter([x, x], [ofi[0], ofi[1]], color = 'gold', marker = "x", label = "Offline CI" if legend_flag else "")
    plt.vlines(x, ofi[0], ofi[1], color = 'gold')
    x += 0.2
    plt.scatter([x, x], [oni[0], oni[1]], color = 'red', marker = "^", label = "Online CI" if legend_flag else "")
    plt.vlines(x, oni[0], oni[1], color = 'red')
    x += 0.6
    legend_flag = False

plt.hlines(theta, 0.8, 10.2, color = 'black', linestyle='dashed')
plt.ylim((2, 17))
plt.legend(loc = 'lower right')
plt.xlabel("Trial #")
plt.savefig("anova_ci_width.svg")
plt.show()

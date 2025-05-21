import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import sys
import multiprocessing as mp

np.random.seed(0)


n = 10
boot_iters = 100
mc_iters = 1000

theta = 10
sigma_alpha = 1/n
sigma_beta = 1/(2*n)
sigma = 1

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


def mc_iteration(mc_iter, nom_coverage):
    np.random.seed(mc_iter)
    exact_coverage = 0
    online_gue_coverage = 0
    offline_gue_coverage = 0

    xs = np.array([theta + st.gamma.rvs(a = 1, loc = -sigma_alpha, scale = sigma_alpha) + st.gamma.rvs(a = 1, loc = -sigma_beta, scale = sigma_beta) + st.gamma.rvs(a = 1, loc = -sigma, scale = sigma) for _ in range(n)])
    thetahat = np.mean(xs)

    distances = []
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

    return (exact_coverage, online_gue_coverage, offline_gue_coverage)

exact_coverages = []
online_coverages = []
offline_coverages = []
for nom_coverage in np.linspace(0, 100, 100)[80:]:
    with mp.Pool(4) as p:
        coverages = p.starmap(mc_iteration, [(i, nom_coverage) for i in range(1000)])

    exact_coverage = np.mean([e for (e, on, off) in coverages])
    online_coverage = np.mean([on for (e, on, off) in coverages])
    offline_coverage = np.mean([off for (e, on, off) in coverages])
    exact_coverages.append(exact_coverage)
    online_coverages.append(online_coverage)
    offline_coverages.append(offline_coverage)
    print(nom_coverage)
    print(exact_coverages)
    print(online_coverage)
    print(offline_coverages)
    print()

# Final Results
"""
nom_coverages = np.linspace(0, 1, 100)[80:]
bootstrap_coverages = [np.float64(0.709), np.float64(0.715), np.float64(0.721), np.float64(0.73), np.float64(0.736), np.float64(0.741), np.float64(0.753), np.float64(0.762), np.float64(0.769), np.float64(0.779), np.float64(0.79), np.float64(0.802), np.float64(0.812), np.float64(0.827), np.float64(0.834), np.float64(0.849), np.float64(0.86), np.float64(0.869), np.float64(0.885), np.float64(0.906)]
online_coverages = [1.0] * len(nom_coverages)
offline_coverages = [np.float64(0.881), np.float64(0.886), np.float64(0.895), np.float64(0.9), np.float64(0.906), np.float64(0.912), np.float64(0.916), np.float64(0.922), np.float64(0.929), np.float64(0.943), np.float64(0.953), np.float64(0.957), np.float64(0.968), np.float64(0.976), np.float64(0.978), np.float64(0.985), np.float64(0.989), np.float64(0.993), np.float64(0.996), np.float64(1.0)]
"""

plt.title("Coverage of Two-Way Mean")
plt.scatter(nom_coverages, bootstrap_coverages, color = "blue", label = "Bootstrapped CS")
plt.scatter(nom_coverages, online_coverages, color = "red", marker = "^", label = "Online GUe CS")
plt.scatter(nom_coverages, offline_coverages, color = "gold", marker = "+", label = "Offline GUe CS")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.legend()
plt.savefig("anova_coverage_gamma.svg")
plt.show()

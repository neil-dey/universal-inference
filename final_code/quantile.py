import numpy as np
import scipy.stats as st
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import sys
import time

np.random.seed(0)

QUANTILE = 0.841345
theta = 0
n = 100
boot_iters = 100
mc_iters = 500

def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.6+
    count = len(it)
    start = time.time() # time estimate start
    def show(j):
        x = int(size*j/count)
        # time estimate calculation and string
        remaining = ((time.time() - start) / j) * (count - j)
        mins, sec = divmod(remaining, 60) # limited to minutes
        time_str = f"{int(mins):02}:{sec:03.1f}"
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}", end='\r', file=out, flush=True)
    show(0.1) # avoid div/0
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def loss(x, theta, tau = QUANTILE):
    return (x-theta)*(tau - (x < theta))

def emp_risk(theta, xs, tau = QUANTILE):
    return np.inf if theta < 0 else np.mean([loss(x, theta, tau) for x in xs])

def log_gue_over_omega_fn(data, true_value):
    data_train = data[:len(data)//2]
    data_test = data[len(data)//2:]
    return -1 * len(data_test) * (emp_risk(max([0, np.quantile(data_train, QUANTILE)]), data_test) - emp_risk(true_value, data_test))

def offline_gue(data, true_value, alpha):
    bootstrap_iters = 100
    omegas = np.linspace(0, 10, 100)[1:]
    coverages = np.zeros(len(omegas))

    data_train = data[:len(data)//2]
    data_test = data[len(data)//2:]

    for boot_iter in range(bootstrap_iters):
        boot_data = data_train[np.random.choice(len(data_train), size = len(data_train), replace = True)]
        lgoo = log_gue_over_omega_fn(boot_data, max([0, np.quantile(data_train, QUANTILE)]))
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
    omegas = np.linspace(0, 1, num = 1000)[1:]

    omega_hats = np.zeros(len(data))
    for n in range(0, len(data)):
        coverages = np.zeros(len(omegas))
        for _ in range(boot_iters):
            boot_data = data[np.random.choice(n+1, n+1, replace = True)]


            boot_erms = [max([0, np.quantile(boot_data[0: i+1], QUANTILE)]) for i in range(len(boot_data))]
            boot_erms = [0] + boot_erms
            boot_mu = max([0, np.quantile(data[:n+1], QUANTILE)])

            excess_losses = [(thetahat - x)**2 - (boot_mu - x)**2 for (thetahat, x) in zip(boot_erms, boot_data)]
            log_gue_over_omega = sum([-1*excess_losses[i] for i in range(n-1)])
            for idx, omega in enumerate(omegas):
                log_gue = omega*log_gue_over_omega
                coverages[idx] += log_gue < np.log(1/alpha)
        coverages /= boot_iters
        omega_hats[n] = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]

    print(omega_hats)
    erms = [max([0, np.quantile(data[0: i+1], QUANTILE)]) for i in range(len(data))]
    erms = [0] + erms
    excess_losses = [(thetahat - x)**2 - (true_value - x)**2 for (thetahat, x) in zip(erms, data)]
    return sum([-1*omega_hat * excess_loss for (omega_hat, excess_loss) in zip(omega_hats, excess_losses)]) < np.log(1/alpha)

nom_coverages = np.linspace(1, 100, num=100)[80:]
exact_coverages = []
online_gue_coverages = []
offline_gue_coverages = []
for nom_coverage in nom_coverages:
    continue
    print(nom_coverage)
    exact_coverage = 0
    online_gue_coverage = 0
    offline_gue_coverage = 0
    #for mc_iter in progressbar(range(mc_iters), str(np.round(nom_coverage, 0)) + " "):
    for mc_iter in range(mc_iters):
        xs = st.norm.rvs(loc = -1, scale = 1, size = n)
        thetahat = max(0, np.quantile(xs, QUANTILE))

        distances = []
        for _ in range(boot_iters):
            indices = np.random.choice(n, n)
            boot_xs = xs[indices]
            boot_thetahat = max(0, np.quantile(boot_xs, QUANTILE))
            distances.append(abs(boot_thetahat - thetahat))

        #print(abs(thetahat - theta),  np.percentile(distances, nom_coverage))
        if(abs(thetahat - theta) < np.percentile(distances, nom_coverage)):
            exact_coverage += 1

        if online_gue(xs, theta, 1-nom_coverage/100):
            online_gue_coverage += 1


        """
        if offline_gue(xs, theta, 1-nom_coverage/100):
            offline_gue_coverage += 1
        """

    exact_coverages.append(exact_coverage/mc_iters)
    online_gue_coverages.append(online_gue_coverage/mc_iters)
    offline_gue_coverages.append(offline_gue_coverage/mc_iters)
    print(exact_coverages)
    print(online_gue_coverages)
    print(offline_gue_coverages)
    print()


# Quantile coverages
nom_coverages = [0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

exact_coverages = [0.552, 0.578, 0.594, 0.594, 0.64, 0.672, 0.704, 0.704, 0.72, 0.72, 0.752, 0.752, 0.764, 0.752, 0.78, 0.812, 0.848, 0.876, 0.906, 0.922]
online_gue_coverages = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
offline_gue_coverages = [0.926, 0.938, 0.938, 0.938, 0.938, 0.938, 0.938, 0.954, 0.954, 0.954, 0.954, 0.954, 0.96, 0.954, 0.954, 0.954, 0.956, 0.97, 0.986, 0.986]

plt.scatter(nom_coverages, exact_coverages, color = "blue", label = "Bootstrapped CS")
plt.scatter(nom_coverages, online_gue_coverages, color = "red", marker = "^", label = "Online GUe CS")
plt.scatter(nom_coverages, offline_gue_coverages, color = "gold", marker = "+", label = "Offline GUe CS")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.legend()
plt.savefig("quantile.svg")
plt.show()

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

mode = sys.argv[1]
if mode != "off" and mode != "on":
    print("Bad argument: Provide \"on\" or \"off\"")
    exit()

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

def emprisk(theta, xs, tau = QUANTILE):
    return np.inf if theta < 0 else sum([loss(x, theta, tau) for x in xs])

def online_loss(x, theta, tau = QUANTILE):
    return np.inf if theta < 0 else (x-theta)*(tau - (x < theta))


def _gibbs(true_theta, data, alpha, omega):
    if mode == "off":
        train_data = data[0:len(data)//2]
        test_data = data[len(data)//2:]
        thetahat = max(0, np.quantile(train_data, QUANTILE))

        log_gue = -1*omega * (emprisk(thetahat, test_data) - emprisk(true_theta, test_data))
    elif mode == "on":
        thetahats = np.zeros(len(data) + 1)
        for idx, x in enumerate(data, start = 1):
            thetahats[idx] = max(0, np.quantile(data[0:idx], QUANTILE))

        log_gue = -1*omega*sum([online_loss(x, thetahat) - online_loss(x, true_theta) for (x, thetahat) in zip(data, thetahats)])

    return log_gue < np.log(1/alpha)

def gibbs(true_theta, data, alpha):
    thetahat = max(0, np.mean(data))# minimize_scalar(lambda theta: emprisk(theta, data), bounds = (0, 10)).x
    coverages = []
    omegas = np.linspace(0, 100, num=100)[1:]
    for omega in omegas:
        coverage = 0
        for _ in range(boot_iters):
            boot_data = np.random.choice(data, size = len(data), replace = True)
            if _gibbs(thetahat, boot_data, alpha, omega):
                coverage += 1
        coverage /= boot_iters
        coverages.append(coverage)

    omega = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]
    #print([(np.round(omega, 2), coverage) for (omega, coverage) in zip(omegas, coverages)])
    print("   " , omega)
    return _gibbs(true_theta, data, alpha, omega)



nom_coverages = np.linspace(1, 100, num=100)[79:-1]
exact_coverages = []
gue_coverages = []
for nom_coverage in nom_coverages:
    continue
    print(nom_coverage)
    exact_coverage = 0
    gue_coverage = 0
    for mc_iter in progressbar(range(mc_iters), str(np.round(nom_coverage, 0)) + " "):
    #for mc_iter in range(mc_iters):
        xs = st.norm.rvs(loc = 1, scale = 1, size = n)
        thetahat = max(0, np.quantile(xs, QUANTILE))

        distances = []
        for _ in range(boot_iters):
            indices = np.random.choice(n, n)
            boot_xs = xs[indices]
            boot_thetahat = max(0, np.quantile(boot_xs, QUANTILE))
            distances.append(abs(boot_thetahat - thetahat))

        if(abs(thetahat - theta) < np.percentile(distances, nom_coverage)):
            exact_coverage += 1

        if(gibbs(theta, xs, 1-nom_coverage/100)):
            gue_coverage += 1

    exact_coverages.append(exact_coverage/mc_iters)
    gue_coverages.append(gue_coverage/mc_iters)
    print(exact_coverages)
    print(gue_coverages)

# Mean coverages
"""
nom_coverages = [x/100 for x in nom_coverages]
exact_coverages = [0.554, 0.638, 0.656, 0.656, 0.682, 0.672, 0.716, 0.716, 0.752, 0.75, 0.786, 0.832, 0.846, 0.866, 0.88, 0.914, 0.9, 0.93, 0.96]
offline_gue_coverages = [0.908, 0.856, 0.884, 0.902, 0.908, 0.906, 0.928, 0.936, 0.92, 0.94, 0.962, 0.962, 0.958, 0.964, 0.978, 0.986, 0.984, 0.992, 0.996]
online_gue_coverages = [0.964, 0.954, 0.964, 0.96, 0.95, 0.952, 0.974, 0.97, 0.97, 0.974, 0.984, 0.97, 0.968, 0.958, 0.97, 0.98, 0.974, 0.99, 0.996]
plt.title("Coverage of Restricted Mean")
"""

# Quantile coverages
nom_coverages = [x/100 for x in nom_coverages]

exact_coverages = [0.528,0.536,0.554,0.582,0.602,0.618,0.644,0.668,0.682,0.704, 0.72,0.742,0.762,0.788,0.806,0.828, 0.86,0.876,0.914,0.946]
offline_gue_coverages = [0.92, 0.926,0.932,0.94, 0.946,0.956,0.958,0.964,0.964,0.972,0.974,0.976,0.976, 0.98,0.982,0.984, 0.99, 0.99,0.996,0.996]
online_gue_coverages = [0.968,0.97, 0.974,0.98, 0.978,0.982,0.984,0.984,0.984,0.986,0.986,0.986,0.988,0.992,0.992,0.994,0.996, 0.996,0.996,0.998]

plt.scatter(nom_coverages, exact_coverages, color = "blue", label = "Bootstrapped CS")
plt.scatter(nom_coverages, online_gue_coverages, color = "red", marker = "^", label = "Online GUe CS")
plt.scatter(nom_coverages, offline_gue_coverages, color = "gold", marker = "+", label = "Offline GUe CS")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.legend()
plt.savefig("quantile.svg")
plt.show()

import numpy as np
import scipy.stats as st
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import sys
import time

np.random.seed(0)


n = 10
boot_iters = 100
mc_iters = 1000

theta = 10
sigma_alpha = 1/n
sigma_beta = 1/(2*n)
sigma = 1

nom_coverage = int(sys.argv[1])

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

def loss(x, theta):
    return (x-theta)**2

def emprisk(theta, xs):
    return sum([loss(x, theta) for x in xs])

def _gibbs(true_theta, data, alpha, omega, mode):
    if mode == "off":
        train_data = data[0:len(data)//2]
        test_data = data[len(data)//2:]
        thetahat = np.mean(train_data)
        log_gue = -1*omega * (emprisk(thetahat, test_data) - emprisk(true_theta, test_data))

    elif mode == "on":
        thetahats = np.zeros(len(data) + 1)
        running_sum = 0
        for idx, x in enumerate(data, start = 1):
            running_sum += x
            thetahats[idx] = running_sum/idx

        log_gue = -1*omega*sum([loss(x, thetahat) - loss(x, true_theta) for (x, thetahat) in zip(data, thetahats)])

    return log_gue < np.log(1/alpha)

def gibbs(true_theta, data, alpha, mode):
    thetahat = np.mean(data)
    coverages = []
    omegas = np.linspace(0, 3, num=100)[1:]
    for omega in omegas:
        coverage = 0
        for _ in range(boot_iters):
            boot_data = np.random.choice(data, size = len(data), replace = True)
            if _gibbs(thetahat, boot_data, alpha, omega, mode):
                coverage += 1
        coverage /= boot_iters
        coverages.append(coverage)

    omega = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]
    #print([(np.round(omega, 2), coverage) for (omega, coverage) in zip(omegas, coverages)])
    #print("   " , omega, coverages[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])])
    return _gibbs(true_theta, data, alpha, omega, mode)




exact_coverage = 0
online_gue_coverage = 0
offline_gue_coverage = 0
for mc_iter in progressbar(range(mc_iters), str(np.round(nom_coverage, 0)) + " "):
#for mc_iter in range(mc_iters):
    continue
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

print("Nominal Coverage:", nom_coverage/100, "\nBootstrap Coverage:", exact_coverage/mc_iters, "\nOnline Coverage:", online_gue_coverage/mc_iters, "\nOffline Coverage:", offline_gue_coverage/mc_iters, "\n", flush=True)



# Normal coverages
"""
nom_coverages = [0.94, 0.92, 0.98, 0.93, 0.97, 0.9, 0.99, 0.95, 0.96, 0.91, 0.83, 0.89, 0.85, 0.88, 0.8, 0.87, 0.82, 0.81, 0.84, 0.86]
exact_coverages = [0.882, 0.861, 0.923, 0.875, 0.913, 0.84, 0.938, 0.891, 0.899, 0.851, 0.763, 0.829, 0.783, 0.818, 0.736, 0.806, 0.754, 0.745, 0.773, 0.797]
online_gue_coverages = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
offline_gue_coverages = [0.984, 0.975, 0.993, 0.979, 0.99, 0.974, 0.999, 0.984, 0.988, 0.974, 0.962, 0.973, 0.966, 0.973, 0.955, 0.972, 0.958, 0.956, 0.964, 0.97]
"""
nom_coverages = [0.92, 0.9, 0.91, 0.94, 0.87, 0.93, 0.88, 0.86, 0.89, 0.85, 0.82, 0.8, 0.84, 0.83, 0.81, 0.95, 0.97, 0.99, 0.98, 0.96]
bootstrap_coverages = [0.806, 0.792, 0.801, 0.827, 0.767, 0.817, 0.778, 0.758, 0.787, 0.749, 0.727, 0.717, 0.744, 0.733, 0.722, 0.84, 0.875, 0.904, 0.888, 0.855]
online_coverages = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
offline_coverages = [0.949, 0.928, 0.938, 0.96, 0.909, 0.952, 0.917, 0.909, 0.923, 0.901, 0.884, 0.873, 0.898, 0.887, 0.875, 0.967, 0.984, 0.999, 0.996, 0.975]

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

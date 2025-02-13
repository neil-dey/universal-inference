import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("error")

np.random.seed(0)
boot_iters = 100

def p(x, theta):
    try:
        return 1/(1+np.exp(-theta[0] - theta[1]*x))
    except RuntimeWarning:
        print((theta[0], theta[1]))
        exit()

def loss(x, y, theta):
    try:
        return (1 + np.exp(y * (theta[0] + theta[1] * x)))**-2
    except RuntimeWarning:
        print(x, y, theta[0], theta[1])
        exit()

def emprisk(theta, xs, ys):
    return sum([loss(x, y, theta) for (x, y) in zip(xs, ys)])
def emprisk_jac(theta, xs, ys):
    try:
        return sum([-2*(1 + np.exp(y * (theta[0] + theta[1] * x)))**-3 * np.exp(y * (theta[0] + theta[1]*x)) * y * np.array([1, x]) for (x, y) in zip(xs, ys)])
    except RuntimeWarning:
        print(x, y, theta[0], theta[1])
        exit()

def precompute(true_theta, data):
    thetahats = []
    xs = []
    ys = []
    for (x, y) in data:
        xs.append(x)
        ys.append(y)
        thetahat = minimize(emprisk, x0 = np.array([1, 1]), args = (xs, ys), method = "L-BFGS-B", jac = emprisk_jac, bounds = ((-100, 100), (-100, 100))).x
        thetahats.append(thetahat)

    return -1*sum([loss(x, y, thetahat) - loss(x, y, true_theta) for (x, y, thetahat) in zip(xs, ys, thetahats)])

def _gibbs(true_theta, data, alpha, omega):
    thetahats = []
    xs = []
    ys = []
    for (x, y) in data:
        xs.append(x)
        ys.append(y)
        thetahat = minimize(emprisk, x0 = np.array([1, 1]), args = (xs, ys), method = "L-BFGS-B", jac = emprisk_jac, bounds = ((-100, 100), (-100, 100))).x
        thetahats.append(thetahat)

    log_gue = -1*omega*sum([loss(x, y, thetahat) - loss(x, y, true_theta) for (x, y, thetahat) in zip(xs, ys, thetahats)])

    return log_gue < np.log(1/alpha)

def gibbs(true_theta, data, alpha):
    thetahat = minimize(emprisk, x0 = np.array([1, 1]), args = (xs, ys), method = "L-BFGS-B", jac = emprisk_jac, bounds = ((-100, 100), (-100, 100))).x
    omegas = np.linspace(0, 2, num=1000)[1:]
    coverages = np.zeros(len(omegas))
    for _ in range(boot_iters):
        boot_data = data[np.random.choice(len(data), len(data), replace = True)]
        p = precompute(thetahat, boot_data)
        for idx, omega in enumerate(omegas):
            if omega * p < np.log(1/alpha):
                coverages[idx] += 1

    coverages /= boot_iters
    omega = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]
    return omega # _gibbs(true_theta, data, alpha, omega)

xs = list(st.norm.rvs(0, 1, 2))
ys = [1 if np.random.rand() < p(x, [1, 1]) else -1 for x in xs]
omegas = []
for n in range(48):
    continue
    new_x = st.norm.rvs()
    xs.append(new_x)
    if np.random.rand() < p(new_x, [1, 1]):
        ys.append(1)
    else:
        ys.append(-1)


    omegas.append(gibbs(np.array([1, 1]), np.array([z for z in zip(xs, ys)]), 0.05))
    print(n+3)
    print(omegas)
    print()

with open("./savage_logistic/omega_paths.txt", "r") as f:
    omega_paths = eval(f.readline())
with open("./savage_logistic/coverages.txt", "r") as f:
    coverages = eval(f.readline())

omega_paths = np.array(omega_paths)
omega_paths = np.mean(omega_paths, axis = 0)
print(omega_paths)
coverages = np.array(coverages)
coverages = np.mean(coverages, axis = 0)
print(coverages)

fig, ax1 = plt.subplots()
ax1.set_xlabel("n")
ax1.set_ylabel("ω")
ax1.plot([n+3 for n in range(50)], omega_paths, color = 'blue', label = "ω")

ax2 = ax1.twinx()
ax2.set_ylabel("Coverage")
ax2.plot([n+3 for n in range(50)], coverages, color = 'red', label = "Coverage")
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

ax1.plot([n+3 for n in range(50)], [1 for n in range(50)], color = 'blue', linestyle = 'dashed')
ax2.plot([n+3 for n in range(50)], [0.95 for n in range(50)], color = 'red', linestyle = 'dashed')

ax1.set_ylim([0, 1.75])
ax2.set_ylim([0, 1])
plt.savefig("average_online_paths.png")
exit()

for omega_path in omegas:
    plt.plot([n+3 for n in range(50)], omega_path)
plt.savefig("learning_rate_path.png")

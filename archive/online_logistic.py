import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#import warnings
#warnings.filterwarnings("error")

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

def emprisk_hess(theta, xs, ys):
    return sum([6*(1 + np.exp(y * (theta[0] + theta[1] * x)))**-4 * np.exp(2*y * (theta[0] + theta[1]*x)) * y**2 * np.array([[1], [x]]) @ np.array([[1, x]]) - 2*(1 + np.exp(y * (theta[0] + theta[1] * x)))**-3 * np.exp(y * (theta[0] + theta[1]*x)) * y**2 * np.array([[1], [x]]) @ np.array([[1, x]]) for (x, y) in zip(xs, ys)])

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
    global GIBBS_PRECOMPUTE
    if GIBBS_PRECOMPUTE is None:
        thetahats = []
        xs = []
        ys = []
        for (x, y) in data:
            xs.append(x)
            ys.append(y)
            thetahat = minimize(emprisk, x0 = np.array([1, 1]), args = (xs, ys), method = "L-BFGS-B", jac = emprisk_jac, bounds = ((-100, 100), (-100, 100))).x
            thetahats.append(thetahat)

        GIBBS_PRECOMPUTE = -1* sum([loss(x, y, thetahat) - loss(x, y, true_theta) for (x, y, thetahat) in zip(xs, ys,           thetahats)])
    log_gue = omega*GIBBS_PRECOMPUTE

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
    new_x = st.norm.rvs()
    xs.append(new_x)
    if np.random.rand() < p(new_x, [1, 1]):
        ys.append(1)
    else:
        ys.append(-1)


    omegas.append(gibbs(np.array([1, 1]), np.array([z for z in zip(xs, ys)]), 0.05))
    #thetahat = minimize(emprisk, x0 = np.array([1, 1]), args = (xs, ys), method = "L-BFGS-B", jac = emprisk_jac, bounds = ((-100, 100), (-100, 100))).x
    print(n+3)
    print(omegas)
    print()

exit()

omegas = np.linspace(0, 10, num = 100)
coverages = np.zeros(100)
max_n = 50
for boot_iter in range(100):
    xs = []
    ys = []
    for n in range(max_n):
        new_x = st.norm.rvs()
        xs.append(new_x)
        if np.random.rand() < p(new_x, [1, 1]):
            ys.append(1)
        else:
            ys.append(-1)
    GIBBS_PRECOMPUTE = None
    for idx, omega in enumerate(np.linspace(0, 10, num = 100)):
        coverages[idx] += _gibbs(np.array([1, 1]), np.array([z for z in zip(xs, ys)]), 0.05, omega)
    print(coverages/(boot_iter+1))
coverages /= 100
exit()


coverages = [1., 1., 1., 1., 1., 1., 1., 1., 0.99, 0.96, 0.93, 0.89, 0.8, 0.69, 0.6, 0.53, 0.43, 0.35, 0.3, 0.2, 0.14, 0.05, 0.04, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ]
plt.plot(omegas, coverages, label = 'n = 100')
coverages = [1., 1., 1., 1., 1., 1., 1., 1., 0.99, 0.99, 0.99, 0.99, 0.99, 0.98, 0.98, 0.97, 0.96, 0.91, 0.88, 0.83, 0.8, 0.78, 0.77, 0.75, 0.74, 0.71, 0.68, 0.62, 0.55, 0.52, 0.49, 0.46, 0.46, 0.41, 0.41, 0.39, 0.35, 0.34, 0.33, 0.32, 0.31, 0.31, 0.29, 0.29, 0.29, 0.28, 0.28, 0.26, 0.25, 0.22, 0.21, 0.18, 0.18, 0.16, 0.16, 0.15, 0.14, 0.14, 0.13, 0.11, 0.1, 0.1, 0.09, 0.09, 0.09, 0.09, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
plt.plot(omegas, coverages, label = 'n = 10')
coverages = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.98, 0.98, 0.98, 0.97, 0.97, 0.97, 0.96, 0.95, 0.94, 0.93, 0.91, 0.9, 0.88, 0.87, 0.86, 0.86, 0.85, 0.85, 0.85, 0.85, 0.84, 0.83, 0.83, 0.81, 0.81, 0.81, 0.81, 0.8, 0.8, 0.79, 0.78, 0.77, 0.75, 0.75, 0.7, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.67, 0.66, 0.66, 0.65, 0.62, 0.62, 0.61, 0.61, 0.6, 0.58, 0.58, 0.56, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.54, 0.54, 0.54, 0.53, 0.52, 0.51, 0.51, 0.5, 0.48, 0.48, 0.48, 0.48]
plt.plot(omegas, coverages, label = 'n = 2')
plt.plot(omegas, [0.95 for omega in omegas], color = 'black', linestyle = 'dashed')
plt.vlines(1, 0, 1, color = 'black', linestyle = 'dashed')
plt.legend()
plt.savefig("omega_effect.png")
exit()


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

import numpy as np
import scipy.stats as st
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import sys
import time

np.random.seed(0)

theta = 0
n = 10
boot_iters = 100
mc_iters = 1000

def loss(x, theta):
    return (x-theta)**2

def emprisk(theta, xs):
    return sum([loss(x, theta) for x in xs])

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
    omegas = np.linspace(0, 10, num=100)[1:]
    for omega in omegas:
        coverage = 0
        for _ in range(boot_iters):
            boot_data = np.random.choice(data, size = len(data), replace = True)
            if _gibbs(thetahat, boot_data, alpha, omega, mode):
                coverage += 1
        coverage /= boot_iters
        coverages.append(coverage)

    omega = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]
    return _gibbs(true_theta, data, alpha, omega, mode)

nom_coverages = np.linspace(0, 1, num=100)[80:-1] * 100
for nom_coverage in nom_coverages:
    exact_coverage = 0
    online_gue_coverage = 0
    offline_gue_coverage = 0
    catoni_coverage = 0
    for mc_iter in range(mc_iters):
        xs = st.t.rvs(df=3, size = n)
        thetahat = np.mean(xs)
        s2 = st.t.stats(df=3)[1]
        catoni_coverage += catoni(theta, xs, s2,  1-nom_coverage/100)

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

    print("Nominal Coverage:", nom_coverage/100)
    print("Catoni Coverage:", catoni_coverage/mc_iters)
    print("Bootstrap Coverage:", exact_coverage/mc_iters)
    print("Online Coverage:", online_gue_coverage/mc_iters)
    print("Offline Coverage:", offline_gue_coverage/mc_iters)
    print()

exit()

# Final Results
nom_coverages = [0.94, 0.91, 0.98, 0.96, 0.93, 0.95, 0.97, 0.92, 0.89, 0.9, 0.83, 0.82, 0.8, 0.81, 0.87, 0.86, 0.85, 0.88, 0.84]
exact_coverages = [0.897, 0.859, 0.944, 0.921, 0.883, 0.908, 0.931, 0.871, 0.836, 0.848, 0.763, 0.755, 0.729, 0.739, 0.811, 0.795, 0.784, 0.821, 0.772]
online_gue_coverages = [0.975, 0.966, 0.991, 0.981, 0.973, 0.978, 0.985, 0.968, 0.963, 0.967, 0.954, 0.952, 0.951, 0.953, 0.961, 0.96, 0.958, 0.962, 0.957]
offline_gue_coverages = [0.964, 0.937, 0.996, 0.982, 0.952, 0.972, 0.99, 0.944, 0.922, 0.929, 0.892, 0.885, 0.879, 0.882, 0.917, 0.904, 0.902, 0.916, 0.898]
catoni_coverages = [1.0, 0.999, 1.0, 1.0, 0.999, 1.0, 1.0, 1.0, 1.0, 0.998, 0.996, 0.999, 0.997, 0.992, 0.998, 0.998, 0.999, 0.997, 0.993]

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

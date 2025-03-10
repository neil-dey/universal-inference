import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(0)

def log_gibbs(theta, data, omega):
    return -omega * sum([loss(x, y, theta) for (x, y) in data])


def MH(data, theta_0, num_iters, omega):
    burn_in_iters = num_iters//10
    x = [theta_0]
    ndim = len(theta_0)
    num_acceptances = 0
    scale = 1
    acceptance_ratio_denom = 0
    for t in range(num_iters):
        if t < burn_in_iters:
            if t % 500 == 0:
                num_acceptances = 0
                acceptance_ratio_denom = 0
            if t % 30 == 0:
                if num_acceptances/(acceptance_ratio_denom+1) < .4:
                    scale *= .75**2
                elif num_acceptances/(acceptance_ratio_denom+1) > .5:
                    scale *= 1.25**2


        x_prime = st.multivariate_normal.rvs(x[t], cov = np.eye(ndim)*scale)
        log_alpha = log_gibbs(x_prime, data, omega) - log_gibbs(x[t], data, omega)
        u = np.random.rand()
        if u < np.exp(log_alpha):
            x.append(x_prime)
            num_acceptances += 1
        else:
            x.append(x[t])
        acceptance_ratio_denom += 1
    #print(scale)
    return x, num_acceptances/acceptance_ratio_denom


def rsafe_bayes(true_theta, data):
    etas = [2**(-i) for i in range(5)]

    min_s = np.inf
    final_eta = etas[0]
    for eta in etas:
        s = 0
        for i in range(len(data) - 1):
            sub_data = data[0:i+1]

            samples , _ = MH(sub_data, np.array([1, 1]), 2000, eta)


            """
            thetas = [[x for (x, y) in samples], [y for (x, y) in samples]]
            fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
            labels = ["theta_0", "theta_1"]
            for i in range(2):
                ax = axes[i]
                ax.plot(thetas[i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)

            axes[-1].set_xlabel("step number");
            plt.show()
            exit()
            """
            r = np.mean([loss(data[i+1][0], data[i+1][1], theta) for theta in samples[1000:]])
            s += r
        if s < min_s:
            min_s = s
            final_eta = eta
    return final_eta



def p(x, theta):
    try:
        return 1/(1+np.exp(-theta[0] - theta[1]*x))
    except RuntimeWarning:
        print((theta[0], theta[1]))
        exit()

def loss(x, y, theta):
    try:
        if y * (theta[0] + theta[1] * x) > 700:
            return 0
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
    boot_iters = 100

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
    return omega #, _gibbs(true_theta, data, alpha, omega)

rsafe_omegas = [[] for i in range(10)]
gue_omegas = [[] for i in range(10)]
rsafe_coverages = np.zeros(10)
gue_coverages = np.zeros(10)

for boot_iter in range(100):
    print(boot_iter)
    xs = list(st.norm.rvs(0, 1, 2))
    ys = [1 if np.random.rand() < p(x, [1, 1]) else -1 for x in xs]
    for n in range(10):
        print("    ", n)
        new_x = st.norm.rvs()
        xs.append(new_x)
        if np.random.rand() < p(new_x, [1, 1]):
            ys.append(1)
        else:
            ys.append(-1)


        rsafe_omega = rsafe_bayes(np.array([1, 1]), np.array([z for z in zip(xs, ys)]))
        rsafe_omegas[n].append(rsafe_omega)

        gue_omega = gibbs(np.array([1, 1]), np.array([z for z in zip(xs, ys)]), 0.1)
        gue_omegas[n].append(gue_omega)

        rsafe_coverages[n] += _gibbs(np.array([1, 1]), np.array([z for z in zip(xs, ys)]), 0.1, rsafe_omega)
        gue_coverages[n] += _gibbs(np.array([1, 1]), np.array([z for z in zip(xs, ys)]), 0.1, gue_omega)
    print(rsafe_coverages/(boot_iter + 1))
    print(gue_coverages/(boot_iter + 1))
    print(rsafe_omegas)
    print(gue_omegas)

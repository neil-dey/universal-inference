from os import sys
import numpy as np
import scipy.stats as st
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from scipy.optimize import brute

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('qt5agg')

np.random.seed(1)

def _gibbs(true_theta, xs, ys, zs, alpha, omega, online):
    if not online:
        train_xs = xs[0:len(ys)//2]
        train_ys = ys[0:len(ys)//2]
        train_zs = zs[0:len(ys)//2]

        test_xs = xs[len(ys)//2:]
        test_ys = ys[len(ys)//2:]
        test_zs = zs[len(ys)//2:]

        thetahat, fval, grid, Jout = brute(emprisk, ((-2, 3), (-2, 6)), Ns = 100, args = (train_xs , train_ys, train_zs), workers = -1, full_output=True, finish = None)
        i_s, j_s = np.where(Jout <= fval)
        candidate_thetahats = [[grid[0,i,j], grid[1,i,j]] for i, j in zip(i_s, j_s)]
        thetahat = np.array(candidate_thetahats[np.argmin(np.linalg.norm(candidate_thetahats, axis = 1))])

        log_gue = -1*omega * (emprisk(thetahat, test_xs, test_ys, test_zs) - emprisk(true_theta, test_xs, test_ys, test_zs))

    else:
        thetahats = [np.array([0, 0])]
        for i in range(len(ys)-2):
            run_xs = xs[0:i+1]
            run_ys = ys[0:i+1]
            run_zs = zs[0:i+1]

            thetahat, fval, grid, Jout = brute(emprisk, ((-2, 2), (-2, 5)), Ns = 100, args = (run_xs, run_ys, run_zs), workers = -1, full_output=True, finish = None)
            i_s, j_s = np.where(Jout <= fval)
            candidate_thetahats = [[grid[0,i,j], grid[1,i,j]] for i, j in zip(i_s, j_s)]
            thetahat = np.array(candidate_thetahats[np.argmin(np.linalg.norm(candidate_thetahats, axis = 1))])
            thetahats.append(thetahat)

        log_gue = -1*omega*sum([loss(x, y, z, thetahat) - loss(x, y, z, true_theta) for (x, y, z, thetahat) in zip(xs, ys, zs, thetahats)])

    return log_gue < np.log(1/alpha)

def gibbs(true_theta, thetahat, xs, ys, zs, alpha, online):
    coverages = []
    """
    omegas = np.linspace(0, 10, num=100)[1:]
    for omega in omegas:
        coverage = 0
        for _ in range(boot_iters):
            indices = np.random.choice(n, n)
            boot_xs = xs[indices]
            boot_ys = ys[indices]
            boot_zs = zs[indices]
            if _gibbs(thetahat, boot_xs, boot_ys, boot_zs, alpha, omega, online):
                coverage += 1
        coverage /= boot_iters
        coverages.append(coverage)
    omega = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]
    print([(np.round(omega, 2), np.round(coverage)) for (omega, coverage) in zip(omegas, coverages)])
    print("   " , omega)
    """
    omega = 5
    return _gibbs(true_theta, xs, ys, zs, alpha, omega, online)

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def loss(x, y, z, theta):
    return (1 - y * sign(x - theta @ z))/2

def emprisk(theta, xs, ys, zs):
    return sum([loss(x, y, z, theta) for (x, y, z) in zip(xs, ys, zs)])

theta = np.array([0, 1])
#risk_minimizer = np.array([0, 0])
risk_minimizer = np.array([0, 0.13])
n = 10
mc_iters = 100
boot_iters = 30


nom_coverage = int(sys.argv[1])
bootstrap_coverage = 0
online_coverage = 0
offline_coverage = 0
distances = []
thetahats = []
for mc_iter in range(mc_iters):
    continue
    print(mc_iter)
    # Generate data
    xs = []
    ys = []
    zs = []

    for _ in range(n):
        z = np.hstack([1, st.beta.rvs(2, 2)])

        """
        x = (np.random.rand() - 0.5)**3 * theta @ z

        t = x/(theta@z)
        y = 1 if np.random.rand() < np.sign(t) * np.abs(t)**(1/3) + 0.5 else -1
        """

        t = np.random.rand()
        if t < 0.5:
            x = 0
        else:
            x = st.uniform.rvs(loc = 0, scale = theta@z)
        def cdf(x, theta, z):
            if x < 0:
                return 0
            elif x < theta @ z:
                return 1/2 + x/(2*theta @ z)
            return 1
        y = 1 if np.random.rand() < cdf(x, theta, z) else -1

        xs.append(x)
        ys.append(y)
        zs.append(z)

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    # Compute thetahat
    #thetahat = minimize(lambda theta: emprisk(xs, ys, zs, theta), np.array([0,0,0]), method = "powell").x


    thetahat, fval, grid, Jout = brute(emprisk, ((-5, 2), (-5, 5)), Ns = 100, args = (xs , ys, zs), workers = -1, full_output=True, finish = None)
    i_s, j_s = np.where(Jout <= fval)
    candidate_thetahats = [[grid[0,i,j], grid[1,i,j]] for i, j in zip(i_s, j_s)]
    thetahat = np.array(candidate_thetahats[np.argmin(np.linalg.norm(candidate_thetahats, axis = 1))])
    #thetahat = np.array(candidate_thetahats[np.argmin([np.linalg.norm(np.array([0, 1]) - c) for c in candidate_thetahats])])
    """
    i_s, j_s, k_s = np.where(Jout <= fval)
    candidate_thetahats = [[grid[0,i,j,k], grid[1,i,j,k], grid[2, i, j, k]] for i, j, k in zip(i_s, j_s, k_s)]
    #print(candidate_thetahats)
    thetahat = np.array(candidate_thetahats[np.argmin(np.linalg.norm(candidate_thetahats, axis = 1))])
    #print(thetahat)
    """
    thetahats.append(thetahat)

    #Bootstrap resamples
    distances = []
    boot_thetahats = []

    for _ in range(boot_iters):
        indices = np.random.choice(n, n)
        boot_xs = xs[indices]
        boot_ys = ys[indices]
        boot_zs = zs[indices]
        """
        boot_thetahat = brute(emprisk, ((thetahat[0]-2, thetahat[0]+2), (thetahat[1]-2, thetahat[1]+5)), args = (boot_xs, boot_ys, boot_zs), workers = -1)
        boot_thetahats.append(boot_thetahat)
        """
        #boot_thetahat, fval, grid, Jout = brute(emprisk, ((thetahat[0]-2, thetahat[0]+2), (thetahat[1]-2, thetahat[1]+5)), Ns = 100, args = (boot_xs , boot_ys, boot_zs), workers = -1, full_output=True, finish = None)
        boot_thetahat, fval, grid, Jout = brute(emprisk, ((thetahat[0]-5, thetahat[0]+2), (thetahat[1]-5, thetahat[1]+5)), Ns = 100, args = (boot_xs , boot_ys, boot_zs), workers = -1, full_output=True, finish = None)
        i_s, j_s = np.where(Jout <= fval)
        candidate_thetahats = [[grid[0,i,j], grid[1,i,j]] for i, j in zip(i_s, j_s)]
        boot_thetahat = np.array(candidate_thetahats[np.argmin(np.linalg.norm(candidate_thetahats, axis = 1))])

        distances.append(np.linalg.norm(boot_thetahat - thetahat))

    if np.linalg.norm(thetahat - risk_minimizer) < np.percentile(distances, nom_coverage):
        bootstrap_coverage += 1

    offline_coverage += gibbs(theta, thetahat, xs, ys, zs, 1 - nom_coverage/100, False)
    #online_coverage += gibbs(theta, thetahat, xs, ys, zs, 1 - nom_coverage/100, True)

"""
print("Minimizers:")
xs = [x for (x, y) in thetahats]
ys = [y for (x, y) in thetahats]
#zs = [z for (x, y, z) in thetahats]
print(np.min(xs), np.mean(xs), np.max(xs))
plt.hist(xs)
plt.savefig("Theta_0s_jump.png")
plt.clf()
print(np.min(ys), np.mean(ys), np.max(ys))
plt.hist(ys)
plt.savefig("Theta_1s_jump.png")
plt.clf()
"""

"""
print(np.min(zs), np.mean([z for (x, y, z) in thetahats]), np.max(zs))
plt.hist([z for (x, y, z) in thetahats])
plt.savefig("Theta_2s_uniform.png")
plt.clf()
"""

offline_coverage /= mc_iters
print("Nominal Coverage:", nom_coverage, "\nObserved Online Coverage:",online_coverage*100, "\nObserved Offline Coverage:", offline_coverage*100)

nom_coverages = np.linspace(0,100,num=100)[80:99]
print(nom_coverages)
boot_coverages  = [84, 84, 85, 86, 86, 88, 88, 90, 90, 93, 93, 93, 93, 93, 94, 94, 95, 95, 96]
offline_gue_coverages = [52, 52, 52, 52, 52, 52, 52, 59, 59, 58, 58, 57, 57, 57, 57, 70, 70, 70, 98]
plt.title("Coverage of MCID Theta")
plt.scatter(nom_coverages, boot_coverages, color = "blue", label = "Bootstrapped CS")
plt.scatter(nom_coverages, offline_gue_coverages, color = "gold", marker = "+", label = "Offline GUe CS")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.legend()
plt.savefig("mcid.png")
plt.show()

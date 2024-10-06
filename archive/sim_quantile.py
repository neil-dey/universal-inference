import numpy as np
import scipy.stats as st
from sklearn.linear_model import QuantileRegressor
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular
from itertools import product as cartesian_product

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import sys

np.random.seed(0)

YEAR_LOW_LIMIT = 1900
YEAR_HIGH_LIMIT = 1980

release_years = []
ratings = []

tau = 0.01
alphas = [0.01, 0.05, 0.2]
p = 2


years = np.linspace(0, 100)
release_years = []
ratings = []
for year in years:
    for _ in range(st.poisson.rvs(100)):
        ratings.append(10 - 3*year + st.norm.rvs(0, 100))
        release_years.append(year)

# Center + scale covariates
center = np.mean(release_years)
scale = np.var(release_years, ddof = 1)
for idx in range(len(release_years)):
    release_years[idx] = (release_years[idx] - center)/scale

release_years = np.array(release_years)
ratings = np.array(ratings)
n = len(ratings)
print(n)

def emprisk(theta, xs, ys, tau):
    return sum([(y - theta @ np.c_[1, x][0]) * (tau - (y - theta @ np.c_[1, x][0] < 0)) for (x, y) in zip(xs, ys)])

def _gibbs(true_theta, xs, ys, alpha, omega, mode, fix_subsample = False):
    if mode == "off":
        if fix_subsample:
            np.random.seed(0)
        indices = np.random.choice(len(ys), len(ys)//2, replace = False)
        train_xs = xs[indices]
        test_xs = np.delete(xs, indices)
        train_ys = ys[indices]
        test_ys = np.delete(ys, indices)

        qr = quantile_regress(train_xs, train_ys, tau, compute_variances = False)
        thetahat = np.hstack([qr.intercept_, qr.coef_])

        log_gue = -1*omega * (emprisk(thetahat, test_xs, test_ys, tau) - emprisk(true_theta, test_xs, test_ys, tau))
    elif mode == "on":
        print("Not implemented yet")
        exit()
        thetahats = np.zeros(len(data) + 1)
        for idx, x in enumerate(data, start = 1):
            thetahats[idx] = max(0, np.quantile(data[0:idx], QUANTILE))

        log_gue = -1*omega*sum([online_loss(x, thetahat) - online_loss(x, true_theta) for (x, thetahat) in zip(data, thetahats)])

    return log_gue < np.log(1/alpha)

def gibbs_lr(xs, ys, tau, alpha, mode, boot_iters = 100):
    qr = quantile_regress(xs, ys, tau, compute_variances = False)
    thetahat = np.hstack([qr.intercept_, qr.coef_])
    coverages = []
    omegas = np.linspace(0, 50, num=500)[1:]
    for omega in omegas:
        coverage = 0
        for _ in range(boot_iters):
            indices = np.random.choice(len(ys), len(ys))
            boot_xs = xs[indices]
            boot_ys = ys[indices]
            if _gibbs(thetahat, boot_xs, boot_ys, alpha, omega, mode):
                coverage += 1
        coverage /= boot_iters
        coverages.append(coverage)

    omega = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]
    omega_coverages = [(np.round(omega, 2), coverage) for (omega, coverage) in zip(omegas, coverages)]
    return omega, omega_coverages


# My own sus version of the powell estimator. Some differences empirically depending on choice of h_n
def sus_powell_estimator(features, ys, tau, p, alpha, qr):
    n = len(ys)
    h = n**(-1/3) * st.norm.ppf(1 - alpha/2)**(2/3) * ((1.5 * st.norm.ppf(tau)**2)/(2 * st.norm.pdf(st.norm.ppf(tau))**2 + 1))**(1/3)
    while (tau - h < 0) or (tau + h > 1):
        h /= 2
    uhat =  ys - qr.predict(features)
    X_mat = np.c_[np.ones(n), features]
    h = (st.norm.ppf(tau + h) - st.norm.ppf(tau - h))*min([np.var(uhat, ddof=1)**0.5, (np.quantile(uhat, 0.75) - np.quantile(uhat, 0.25))/1.34])
    h *= 10
    f = st.norm.pdf(uhat/h)/h
    fxxinv = np.eye(p)
    M = np.linalg.qr(X_mat * f[:,np.newaxis]**0.5).R
    fxxinv = solve_triangular(M, fxxinv)
    fxxinv = fxxinv @ fxxinv.T
    var = tau * (1-tau) * fxxinv @ X_mat.T @ X_mat @ fxxinv
    return var


def rq_estimator(features, ys, tau, p, alpha, qr):
    # Powell sandwich estimator---using same algorithm as rq package in R
    n = len(ys)
    h = n**(-1/3) * st.norm.ppf(1 - alpha/2)**(2/3) * ((1.5 * st.norm.ppf(tau)**2)/(2 * st.norm.pdf(st.norm.ppf(tau))**2 + 1))**(1/3)
    while (tau - h < 0) or (tau + h > 1):
        h /= 2
    uhat =  ys - qr.predict(features)
    X_mat = np.c_[np.ones(n), features]
    h = (st.norm.ppf(tau + h) - st.norm.ppf(tau - h))*min([np.var(uhat, ddof=1)**0.5, (np.quantile(uhat, 0.75) - np.quantile(uhat, 0.25))/1.34])
    f = st.norm.pdf(uhat/h)/h
    fxxinv = np.eye(p)
    M = np.linalg.qr(X_mat * f[:,np.newaxis]**0.5).R
    fxxinv = solve_triangular(M, fxxinv)
    fxxinv = fxxinv @ fxxinv.T
    var = tau * (1-tau) * fxxinv @ X_mat.T @ X_mat @ fxxinv
    return var

def bootstrap_estimator(features, ys, tau, bootstrap_iters):
    betas = []
    for i in range(bootstrap_iters):
        indices = np.random.choice(len(ys), len(ys))
        boot_features = features[indices]
        boot_ys = ys[indices]

        qr = QuantileRegressor(quantile = tau, alpha = 0).fit(boot_features, boot_ys)
        betas.append(np.hstack([qr.intercept_, qr.coef_]))
    return np.cov(betas, rowvar=False)



def quantile_regress(xs, ys, tau, p = 2, alpha = 0.05, compute_variances = True):
    poly = PolynomialFeatures(degree=p - 1, include_bias=False)
    poly_features = poly.fit_transform(xs.reshape(-1, 1))
    print("Regressing")
    qr = QuantileRegressor(quantile = tau, alpha = 0)
    qr = qr.fit(poly_features, ys)

    if compute_variances:
        var_powell = rq_estimator(poly_features, ratings, tau, p, alpha, qr)
        var_sus = sus_powell_estimator(poly_features, ratings, tau, p, alpha, qr)
        var_bootstrap = bootstrap_estimator(poly_features, ratings, tau, 100)
        return qr, var_powell, var_bootstrap, var_sus

    return qr

#qr = quantile_regress(release_years, ratings, tau, p, compute_variances = False)

#omega, omega_coverages = gibbs_lr(release_years, ratings, tau, alpha, "off")

# Fit a linear regression on omega_coverages to get a learning rate for each alpha (data past 1980)
"""
omegas = [1.26, 3.26, 5.76, 8.26, 10.76]
omega = omegas[int(sys.argv[1])]
intercepts = []
slopes = []
for beta0 in np.linspace(0, 8, num = 150):
    for beta1 in np.linspace(-5, 8, num = 250):
        if _gibbs(np.array([beta0, beta1]), release_years, ratings, alpha, omega, "off"):
            intercepts.append(beta0)
            slopes.append(beta1)
print(alpha, [(x, y) for (x, y) in zip(intercepts, slopes)], flush = True)
print()
"""

"""
# Fit a linear regression on omega_coverages to get a learning rate for each alpha (data pre-1980)
# For alpha = 0.01, we get a negative estimate for omega, so we use the first omega observed with empirical coverage 0.99
omegas = [3.21, 10.1, 53.50]
colors = ['purple', 'cyan', 'red']
markers = ['.', 'x', '+']

for alpha, omega, color, marker in zip(alphas, omegas, colors, markers):
    intercepts = []
    slopes = []
    for beta0 in np.linspace(1, 4, num = 100):
        for beta1 in np.linspace(-25, 30, num = 100):
            if _gibbs(np.array([beta0, beta1]), release_years, ratings, alpha, omega, "off", True):
                intercepts.append(beta0)
                slopes.append(beta1)
    plt.scatter(intercepts, slopes, marker = marker, color = color, label = (str((1-alpha)*100) + "% GUe"))

# Plot the 95% rq confidence set
rq_var = np.array([[0.00875972, 0.1156339 ], [0.1156339,  1.81532241]])
rq_var = np.array([[0.00075062, 0.00019724], [0.00019724, 0.10747174]])

rho = rq_var[0, 1]/np.sqrt(rq_var[0,0] * rq_var[1, 1])
mat = (2**0.5*1.96) * np.sqrt(np.diag(np.diag(rq_var))) @ np.array([[1, -1], [1, 1]]) @ np.diag([(1+rho)**0.5, (1-rho)**0.5])
ts = np.linspace(0, 2*np.pi, 100)
plt.plot([(mat @ np.array([[np.cos(t)], [np.sin(t)]]))[0] + qr.intercept_ for t in ts], [(mat @ np.array([[np.cos(t)], [np.sin(t)]]))[1] + qr.coef_[0] for t in ts], color = 'black', linestyle = 'solid', label = '95% rq')

# Plot the 95% powell confidence set
powell_var = np.array([[ 0.25555184,  2.30358895],[ 2.30358895, 63.41877595]])
powell_var = np.array([[ 0.2561472,   1.50325091], [ 1.50325091, 11.38303892]])
rho = powell_var[0, 1]/np.sqrt(powell_var[0,0] * powell_var[1, 1])
mat = (2**0.5*1.96) * np.sqrt(np.diag(np.diag(powell_var))) @ np.array([[1, -1], [1, 1]]) @ np.diag([(1+rho)**0.5, (1-rho)**0.5])
ts = np.linspace(0, 2*np.pi, 100)
plt.plot([(mat @ np.array([[np.cos(t)], [np.sin(t)]]))[0] + qr.intercept_ for t in ts], [(mat @ np.array([[np.cos(t)], [np.sin(t)]]))[1] + qr.coef_[0] for t in ts], color = 'black', linestyle = 'dashed', label = '95% Fixed')

# Plot the 95% bootstrap confidence set
boot_var = np.array([[1.89782747, 0.11704016], [0.11704016, 1.72349186]])
boot_var = np.array([[ 0.13783966, -0.00068021], [-0.00068021,  0.0090499 ]])
rho = boot_var[0, 1]/np.sqrt(boot_var[0,0] * boot_var[1, 1])
mat = (2**0.5*1.96) * np.sqrt(np.diag(np.diag(boot_var))) @ np.array([[1, -1], [1, 1]]) @ np.diag([(1+rho)**0.5, (1-rho)**0.5])
ts = np.linspace(0, 2*np.pi, 100)
plt.plot([(mat @ np.array([[np.cos(t)], [np.sin(t)]]))[0] + qr.intercept_ for t in ts], [(mat @ np.array([[np.cos(t)], [np.sin(t)]]))[1] + qr.coef_[0] for t in ts], color = 'black', linestyle = 'dashdot', label = '95% bootstrap')

plt.xlabel("Intercept")
plt.ylabel("Slope")
plt.legend()
#plt.savefig("contour_plot.svg")
plt.show()
plt.clf()
"""

qr, var_powell, var_bootstrap, var_sus = quantile_regress(release_years, ratings, tau, p)
plt.scatter([x * scale + center for x in release_years], ratings, marker = '.')
xs = np.linspace(min(release_years), max(release_years), num = 100)
poly = PolynomialFeatures(degree=p - 1, include_bias=False)
poly_xs = poly.fit_transform(xs.reshape(-1, 1))
plt.plot([x * scale + center for x in xs], [qr.predict([x]) for x in poly_xs], color = 'black', marker = '.')

var = var_powell
print(var)
plt.plot([x * scale + center for x in xs], [qr.predict([poly_x]) + 1.96* (np.array([np.hstack([1, poly_x])]) @ var @ np.array([np.hstack([1, poly_x])]).T)[0] for poly_x in poly_xs], color = 'red', linestyle = 'dashed', label = "rq")
plt.plot([x * scale + center for x in xs], [qr.predict([poly_x]) - 1.96* (np.array([np.hstack([1, poly_x])]) @ var @ np.array([np.hstack([1, poly_x])]).T)[0] for poly_x in poly_xs], color = 'red', linestyle = 'dashed')

var = var_sus
print(var)
plt.plot([x * scale + center for x in xs], [qr.predict([poly_x]) + 1.96* (np.array([np.hstack([1, poly_x])]) @ var @ np.array([np.hstack([1, poly_x])]).T)[0] for poly_x in poly_xs], color = 'cyan', linestyle = 'dotted', label = "Fixed")
plt.plot([x * scale + center for x in xs], [qr.predict([poly_x]) - 1.96* (np.array([np.hstack([1, poly_x])]) @ var @ np.array([np.hstack([1, poly_x])]).T)[0] for poly_x in poly_xs], color = 'cyan', linestyle = 'dotted')

var = var_bootstrap
print(var)
plt.plot([x * scale + center for x in xs], [qr.predict([poly_x]) + 1.96* (np.array([np.hstack([1, poly_x])]) @ var @ np.array([np.hstack([1, poly_x])]).T)[0] for poly_x in poly_xs], color = 'purple', linestyle = 'dashdot', label = "Bootstrap")
plt.plot([x * scale + center for x in xs], [qr.predict([poly_x]) - 1.96* (np.array([np.hstack([1, poly_x])]) @ var @ np.array([np.hstack([1, poly_x])]).T)[0] for poly_x in poly_xs], color = 'purple', linestyle = 'dashdot')

#plt.ylim(None, 10)
plt.xlabel("Year")
plt.ylabel("Rating")
plt.legend()
plt.show()
#plt.savefig("quantile_regression.svg")

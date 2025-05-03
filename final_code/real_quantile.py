import numpy as np
import scipy.stats as st
from sklearn.linear_model import QuantileRegressor
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular
from itertools import product as cartesian_product
import multiprocessing as mp

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import sys

np.random.seed(0)

YEAR_LOW_LIMIT = 1800
YEAR_HIGH_LIMIT = 1980
toggle_2000 = True if sys.argv[1] == "True" else False
if toggle_2000:
    YEAR_HIGH_LIMIT = 2000

release_years = []
ratings = []

tau = 0.01
p = 2

with open('anime.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile, delimiter = '\t')
    for row in reader:
        if not row['score']:
            continue
        release_year = int(datetime.strptime(row['start_date'], '%Y-%m-%d %H:%M:%S').strftime('%Y'))
        if release_year >= YEAR_LOW_LIMIT and release_year < YEAR_HIGH_LIMIT:
            release_years.append(release_year)
            ratings.append(float(row['score']))

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

def log_gue_over_omega(true_theta, xs, ys, mode, fix_subsample = False):
    if mode == "off":
        if fix_subsample:
            indices = [2*i+1 for i in range(len(ys)//2)]
        else:
            indices = np.random.choice(len(ys), len(ys)//2, replace = False)
        train_xs = xs[indices]
        test_xs = np.delete(xs, indices)
        train_ys = ys[indices]
        test_ys = np.delete(ys, indices)

        qr = quantile_regress(train_xs, train_ys, tau, compute_variances = False)
        thetahat = np.hstack([qr.intercept_, qr.coef_])

        log_gue_over_omega = -1 * (emprisk(thetahat, test_xs, test_ys, tau) - emprisk(true_theta, test_xs, test_ys, tau))
    else:
        print("Not implemented yet")
        exit()

    return log_gue_over_omega


def _gibbs(true_theta, xs, ys, alpha, omega, mode, fix_subsample = False):
    return  omega * log_gue_over_omega(true_theta, xs, ys, mode, fix_subsample) < np.log(1/alpha)

def gibbs_lr(xs, ys, tau, alpha, mode, boot_iters = 1000):
    qr = quantile_regress(xs, ys, tau, compute_variances = False)
    thetahat = np.hstack([qr.intercept_, qr.coef_])
    omegas = np.linspace(0, 50, num=500)[1:]

    training_indices = [2*i + 1 for i in range(len(ys)//2)]

    coverages = np.zeros(len(omegas))
    for _ in range(boot_iters):
        indices = np.random.choice(training_indices, len(training_indices))
        boot_xs = xs[indices]
        boot_ys = ys[indices]
        lgoo =  log_gue_over_omega(thetahat, boot_xs, boot_ys, mode)
        for idx, omega in enumerate(omegas):
            coverages[idx] += (omega * lgoo < np.log(1/alpha))
    coverages /= boot_iters

    omega = omegas[np.argmin([abs(alpha - (1-coverage)) for coverage in coverages])]
    omega_coverages = [(np.round(omega, 2), coverage) for (omega, coverage) in zip(omegas, coverages)]
    return omega, omega_coverages


def rq_estimator(features, ys, tau, p, alpha, qr, modified = False):
    # Powell sandwich estimator---using same algorithm as rq package in R
    n = len(ys)
    h = n**(-1/3) * st.norm.ppf(1 - alpha/2)**(2/3) * ((1.5 * st.norm.ppf(tau)**2)/(2 * st.norm.pdf(st.norm.ppf(tau))**2 + 1))**(1/3)
    while (tau - h < 0) or (tau + h > 1):
        h /= 2

    uhat =  ys - qr.predict(features)
    X_mat = np.c_[np.ones(n), features]
    h = (st.norm.ppf(tau + h) - st.norm.ppf(tau - h))*min([np.var(uhat, ddof=1)**0.5, (np.quantile(uhat, 0.75) - np.quantile(uhat, 0.25))/1.34])
    if modified: # "Fixed mode"
        h *= 10
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

        qr = QuantileRegressor(quantile = tau, alpha = 0).fit(boot_features, ratings)
        betas.append(np.hstack([qr.intercept_, qr.coef_]))
    return np.cov(betas, rowvar=False)



def quantile_regress(xs, ys, tau, p = 2, alpha = 0.05, compute_variances = True):
    poly = PolynomialFeatures(degree=p - 1, include_bias=False)
    poly_features = poly.fit_transform(xs.reshape(-1, 1))
    qr = QuantileRegressor(quantile = tau, alpha = 0)
    qr = qr.fit(poly_features, ys)

    if compute_variances:
        var_powell = rq_estimator(poly_features, ratings, tau, p, alpha, qr)
        var_sus = rq_estimator(poly_features, ratings, tau, p, alpha, qr, modified = True)
        var_bootstrap = bootstrap_estimator(poly_features, ratings, tau, 100)
        return qr, var_powell, var_bootstrap, var_sus

    return qr

qr = quantile_regress(release_years, ratings, tau, p, compute_variances = False)

alpha = 0.05
omega, omega_coverages = gibbs_lr(release_years, ratings, tau, alpha, "off")
print(omega, omega_coverages)
#exit()


# Fit a linear regression on omega_coverages to get a learning rate for each alpha (data pre-1980)
#omegas = [3.21, 10.1, 53.50]
omega = 8.517034068136272
if toggle_2000:
    #omegas = [3.41, 7.31, 24.45] #2000
    omega = 5.110220440881764
#colors = ['purple', 'cyan', 'red']
#markers = ['.', 'x', '+']

#beta0s = [beta0 for beta0 in np.linspace(1, 4.5, num = 120) if (toggle_2000 and not (beta0 < 3 or beta0 > 4.25)) or (not toggle_2000 and beta0 <= 4)]
#beta1s = [beta1 for beta1 in np.linspace(-25, 30, num = 100) if (toggle_2000 and not (beta1 < -3 or beta1 > 20)) or (not toggle_2000 and not (beta1 < -15 or beta1 > 15))]
beta0s = np.linspace(2.5, 4.0, num = 100)
beta1s = np.linspace(-15, 10, num = 100)
if toggle_2000:
    beta0s = np.linspace(3.0, 4.0, num = 67)
    beta1s = np.linspace(-5, 15, num = 80)

args = [(np.array([beta0, beta1]), release_years, ratings, alpha, omega, "off", True) for (beta0, beta1) in cartesian_product(beta0s, beta1s)]

with mp.Pool(4) as pool:
    marks = pool.starmap(_gibbs, args)

intercepts = []
slopes = []
for (idx, beta) in enumerate(cartesian_product(beta0s, beta1s)):
    if marks[idx]:
        intercepts.append(beta[0])
        slopes.append(beta[1])
print(slopes)
print(intercepts)
print()

qr, var_powell, var_bootstrap, var_sus = quantile_regress(release_years, ratings, tau, p)
plt.scatter([x * scale + center for x in release_years], ratings, marker = '.')
xs = np.linspace(min(release_years), max(release_years), num = 100)
poly = PolynomialFeatures(degree=p - 1, include_bias=False)
poly_xs = poly.fit_transform(xs.reshape(-1, 1))
plt.plot([x * scale + center for x in xs], [qr.predict([x]) for x in poly_xs], color = 'black')

var = var_powell
print("rq")
print(var)
plt.plot([x * scale + center for x in xs], [qr.predict([poly_x]) + 1.96* (np.array([np.hstack([1, poly_x])]) @ var @ np.array([np.hstack([1, poly_x])]).T)[0] for poly_x in poly_xs], color = 'red', linestyle = 'dashed', label = "rq")
plt.plot([x * scale + center for x in xs], [qr.predict([poly_x]) - 1.96* (np.array([np.hstack([1, poly_x])]) @ var @ np.array([np.hstack([1, poly_x])]).T)[0] for poly_x in poly_xs], color = 'red', linestyle = 'dashed')

var = var_sus
print("Fixed")
print(var)
plt.plot([x * scale + center for x in xs], [qr.predict([poly_x]) + 1.96* (np.array([np.hstack([1, poly_x])]) @ var @ np.array([np.hstack([1, poly_x])]).T)[0] for poly_x in poly_xs], color = 'cyan', linestyle = 'dotted', label = "Fixed")
plt.plot([x * scale + center for x in xs], [qr.predict([poly_x]) - 1.96* (np.array([np.hstack([1, poly_x])]) @ var @ np.array([np.hstack([1, poly_x])]).T)[0] for poly_x in poly_xs], color = 'cyan', linestyle = 'dotted')

var = var_bootstrap
print("Bootstrap")
print(var)
plt.plot([x * scale + center for x in xs], [qr.predict([poly_x]) + 1.96* (np.array([np.hstack([1, poly_x])]) @ var @ np.array([np.hstack([1, poly_x])]).T)[0] for poly_x in poly_xs], color = 'purple', linestyle = 'dashdot', label = "Bootstrap")
plt.plot([x * scale + center for x in xs], [qr.predict([poly_x]) - 1.96* (np.array([np.hstack([1, poly_x])]) @ var @ np.array([np.hstack([1, poly_x])]).T)[0] for poly_x in poly_xs], color = 'purple', linestyle = 'dashdot')

plt.xlabel("Year")
plt.ylabel("Rating")
plt.legend()
plt.savefig("quantile_regression.svg")

plt.clf()

# NOTE TO PEOPLE WHO ARE MODIFYING THIS CODE
# The variances rq_var, powell_var, and boot_var are all instances from a single run of the above code.
# Really, you ought to just use var_powell, var_boostrap, and var_sus from above.
# But that makes it annoying to generate these plots in 1 run so here's just the hardcoded variances.

# Plot the 95% rq confidence set
rq_var = np.array([[0.00875972, 0.1156339 ], [0.1156339,  1.81532241]])
if toggle_2000:
    rq_var = np.array([[ 5.84120982e-03, -1.07094895e-01], [-1.07094895e-01,  6.05763269e+00]]) #2000
rho = rq_var[0, 1]/np.sqrt(rq_var[0,0] * rq_var[1, 1])
mat = (2**0.5*1.96) * np.sqrt(np.diag(np.diag(rq_var))) @ np.array([[1, -1], [1, 1]]) @ np.diag([(1+rho)**0.5, (1-rho)**0.5])
ts = np.linspace(0, 2*np.pi, 100)
plt.plot([(mat @ np.array([[np.cos(t)], [np.sin(t)]]))[0] + qr.intercept_ for t in ts], [(mat @ np.array([[np.cos(t)], [np.sin(t)]]))[1] + qr.coef_[0] for t in ts], color = 'red', linestyle = 'dashed', label = '95% rq')

# Plot the 95% modified confidence set
powell_var = np.array([[0.01115388, 0.01435639], [0.01435639, 3.93013017]])
if toggle_2000:
    powell_var = np.array([[ 1.81088800e-03, -9.38851193e-05], [-9.38851193e-05,  4.72428417e-01]]) #2000
rho = powell_var[0, 1]/np.sqrt(powell_var[0,0] * powell_var[1, 1])
mat = (2**0.5*1.96) * np.sqrt(np.diag(np.diag(powell_var))) @ np.array([[1, -1], [1, 1]]) @ np.diag([(1+rho)**0.5, (1-rho)**0.5])
ts = np.linspace(0, 2*np.pi, 100)
plt.plot([(mat @ np.array([[np.cos(t)], [np.sin(t)]]))[0] + qr.intercept_ for t in ts], [(mat @ np.array([[np.cos(t)], [np.sin(t)]]))[1] + qr.coef_[0] for t in ts], color = 'purple', linestyle = 'dashdot', label = '95% Fixed')

# Plot the 95% bootstrap confidence set
boot_var = np.array([[ 3.95570142e-03, -8.21251606e-02], [-8.21251606e-02,  6.62865737e+00]])
if toggle_2000:
    boot_var = np.array([[1.04076954e-03, 9.84986950e-03], [9.84986950e-03, 2.99631835e+00]]) #2000
rho = boot_var[0, 1]/np.sqrt(boot_var[0,0] * boot_var[1, 1])
mat = (2**0.5*1.96) * np.sqrt(np.diag(np.diag(boot_var))) @ np.array([[1, -1], [1, 1]]) @ np.diag([(1+rho)**0.5, (1-rho)**0.5])
ts = np.linspace(0, 2*np.pi, 100)
plt.plot([(mat @ np.array([[np.cos(t)], [np.sin(t)]]))[0] + qr.intercept_ for t in ts], [(mat @ np.array([[np.cos(t)], [np.sin(t)]]))[1] + qr.coef_[0] for t in ts], color = 'black', linestyle = 'solid', label = '95% bootstrap')

# Plot the 95% Gue confidence set
plt.scatter(intercepts, slopes, marker = 'x', color = 'cyan', label = (str((1-alpha)*100) + "% GUe"))

plt.xlim(2.5, 4.15)
plt.ylim(-17.5, 20)
plt.xlabel("Intercept")
plt.ylabel("Slope")
plt.legend()
if toggle_2000:
    plt.savefig("contour_plot_2000_only95_uncentered.svg")
else:
    plt.savefig("contour_plot_only95_uncentered.svg")
plt.clf()

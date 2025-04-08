import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import warnings
from sklearn import svm
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("error")


np.random.seed(0)
boot_iters = 5000


def p(x, theta):
    try:
        return 1/(1+np.exp(-theta[0] - theta[1]*x))
    except RuntimeWarning:
        print("Error in p")
        print((theta[0], theta[1]))
        exit()

def loss(x, y, theta):
    hinge = 1 - y*(theta[0] + theta[1]*x)
    if hinge < 0:
        return 0
    return hinge

def emprisk(theta, xs, ys):
    return sum([loss(x, y, theta) for (x, y) in zip(xs, ys)])

def online_gue(true_theta, data, alpha, omega, random = False):
    thetahats = [np.array([0, 0])]
    xs = np.array([])
    ys = []
    min_n = 0
    for (x, y) in data:
        xs = np.append(xs, x)
        ys.append(y)
        # If we haven't observed two different labels yet, you're gonna have a bad time making estimates
        if len(set(ys)) != 2:
            min_n += 1
            continue
        fit = svm.SVC(kernel='linear').fit(xs.reshape(-1, 1), ys)
        thetahat = np.array([fit.intercept_[0], fit.coef_[0][0]])
        thetahats.append(thetahat)

    log_gue = 0
    skip_count = 0
    for (x, y, thetahat) in zip(xs, ys, thetahats):
        # Again, skip calculating GUe-values (effectively setting ω=0) until you have estimates at all
        if skip_count < min_n:
            skip_count+=1
            continue
        if random:
            lr = omega * np.random.rand()*2
        else:
            lr = omega
        log_gue += -1*lr*(loss(x, y, thetahat) - loss(x, y, true_theta))

    return log_gue < np.log(1/alpha)


nonrandom_coverages = np.zeros(20)
random_coverages = np.zeros(20)
bootstrap_coverages = np.zeros(20)
levels = np.linspace(0.8, 0.99, 20)
true_theta = np.array([0, 1])
for idx, conf in enumerate(levels):
    nonrandom_coverage = 0
    random_coverage = 0
    bootstrap_coverage = 0
    #max_n = 100
    for boot_iter in range(boot_iters):
        xs = []
        ys = []
        while True:
            new_x = st.norm.rvs(0, 1)
            xs.append(new_x)
            if np.random.rand() < p(new_x, true_theta):
                ys.append(1)
            else:
                ys.append(-1)
            #Stopping rule
            if sum([x**2 for x in xs]) > 10:
                break
        xs = np.array(xs)
        ys = np.array(ys)

        """
        #Bootstrapped confidence set
        thetahats = []
        logreg = LogisticRegression(penalty=None)
        boot = 0
        while boot < 100:
            indices = np.random.choice(len(xs), len(xs), replace = True)
            boot_xs = xs[indices]
            boot_ys = ys[indices]
            if len(set(boot_ys)) != 2:
                continue
            fit = logreg.fit(boot_xs.reshape(-1, 1), boot_ys)
            thetahats.append(np.array([fit.intercept_[0], fit.coef_[0][0]]))
            boot += 1

        cov_boot = np.cov(thetahats, rowvar = False)
        # Transform
        fit = logreg.fit(xs.reshape(-1, 1), ys)
        logreg_thetahat = np.array([fit.intercept_[0], fit.coef_[0][0]])
        transformed_true_theta = np.linalg.inv(sqrtm(cov_boot)) @ (true_theta - logreg_thetahat)
        sq_dist_to_origin = np.linalg.norm(transformed_true_theta)**2
        pval = 1 - st.chi2.cdf(sq_dist_to_origin, 2)
        if pval > 1 - conf:
            bootstrap_coverage += 1

        print(bootstrap_coverage/(boot_iter + 1))
        """
        nonrandom_coverage += online_gue(true_theta, np.array([z for z in zip(xs, ys)]), 1-conf, 0.45, False)
        random_coverage += online_gue(true_theta, np.array([z for z in zip(xs, ys)]), 1-conf, 0.4, True)

    nonrandom_coverage /= boot_iters
    random_coverage /= boot_iters
    #bootstrap_coverage /= boot_iters
    nonrandom_coverages[idx] = nonrandom_coverage
    random_coverages[idx] = random_coverage
    #bootstrap_coverages[idx] = bootstrap_coverage
    print(conf, nonrandom_coverage, random_coverage, bootstrap_coverage)

plt.scatter(levels, nonrandom_coverages, color = 'blue', label = "ω = 0.45")
plt.scatter(levels, random_coverages, color = 'red', marker = '+', label = "ω ~ Unif(0, 0.8)")
#plt.scatter(levels, bootstrap_coverages, color = 'm', marker = 'x', label = "Bootstrap")
plt.plot(levels, levels, color = 'black', linestyle = 'dashed')
plt.legend()
plt.title("Coverage of SVM support vectors")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.savefig("svm_online.png")

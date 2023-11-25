# WE ARE'NT USING THIS ONE. DON'T LOOK AT IT
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

np.random.seed(0)

# 1 dimensional
mu0 = 5
mu1 = 10
cov = 10000
c = (mu0 + mu1)/2


omega = 0.01

def _gibbs(mu, data0, data1, nom_coverage, omega):
    raw_data = np.array([])
    data = np.empty(shape=(0, 2))
    for theta in data0:
        idx = np.searchsorted(raw_data, theta)
        data = np.insert(data, idx, [theta, 0], axis = 0)
        raw_data = np.insert(raw_data, idx, theta)

    for theta in data1:
        idx = np.searchsorted(raw_data, theta)
        data = np.insert(data, idx, [theta, 1], axis = 0)
        raw_data = np.insert(raw_data, idx, theta)

    num_1s = 0
    c_est = data[0][0]
    for (theta, label) in data:
        if label == 1:
            num_1s += 1
        else:
            num_1s -= 1
            if num_1s <= 0:
                c_est = theta
                num_1s = 0

    def emp_risk_test(theta):
        err_count = np.searchsorted(data1, theta, side="right")
        err_count += len(data0) - np.searchsorted(data0, theta, side="left")
        return err_count
        #return len([x for x in data1_test if x <= theta]) + len([x for x in data0_test if x > theta])

    def T(theta, omega):
        return (omega * (emp_risk_test(c_est) - emp_risk_test(theta)))

    # Check that confidence set contains c
    return T(c, omega) >= np.log(1 - nom_coverage)

def gibbs(mu, data0, data1, nom_coverage):
    boot_iters = 100
    coverages = []
    omegas = np.linspace(0, .1, num = 10)[1:]
    omegas = np.append(omegas, np.linspace(.1, 1, num = 10))
    #omegas = np.append(omegas, np.linspace(1, 100, num = 100))
    #omegas = np.append(omegas, np.linspace(100, 1000, num = 100))
    #omegas = np.append(omegas, np.linspace(1000, 10000000, num = 100))
    for omega in omegas:
        coverage = 0
        for _ in range(boot_iters):
            coverage += _gibbs(mu, np.random.choice(data0, size = len(data0), replace = True), np.random.choice(data1, size = len(data1), replace = True), nom_coverage, omega)
        coverage /= boot_iters
        coverages.append(coverage)

    omega = omegas[np.argmin([abs(nom_coverage - coverage) for coverage in coverages])]
    print("  omega:", omega)
    return _gibbs(mu, data0, data1, nom_coverage, omega)

def mc_iteration(nom_coverage):
    # Exact confidence interval
    exact_coverage = 0
    universal_coverage = 0
    data0 = st.norm.rvs(loc = mu0, scale = cov**0.5, size = 2)
    data1 = st.norm.rvs(loc = mu1, scale = cov**0.5, size = 2)
    n0 = 2
    n1 = 2
    while True:
        # Point estimate for (mu0 + mu1)/2
        c_est = (np.mean(data0) + np.mean(data1))/2

        # (1-\alpha)100% confidence interval for (mu0 + mu1)/2
        z_alpha = st.norm.interval(nom_coverage)[1]
        CI = [c_est - z_alpha* ((n0 + n1)/(4*n0*n1) * cov)**0.5, c_est + z_alpha * ((n0 + n1)/(4*n0*n1) * cov)**0.5]

        # If the confidence interval still includes the origin, collect more data and try again
        if CI[0] <= 0 and 0 <= CI[1]:
            if np.random.rand() < 0.5:
                data0 = np.append(data0, st.norm.rvs(loc = mu0, scale = cov**0.5, size = 1))
                n0 += 1
            else:
                data1 = np.append(data1, st.norm.rvs(loc = mu1, scale = cov**0.5, size = 1))
                n1 += 1
            continue

        # Otherwise, we stop collecting data and check if the confidence interval contains the true (mu0+mu1)/2
        if CI[0] <= c and c <= CI[1]:
            exact_coverage += 1
        break

    # Universal interval
    universal_coverage = gibbs(c, data0, data1, nom_coverage)
    return (exact_coverage, universal_coverage)

    # ERM computation
    raw_data = np.array([])
    data = np.empty(shape=(0, 2))
    for theta in data0:
        idx = np.searchsorted(raw_data, theta)
        data = np.insert(data, idx, [theta, 0], axis = 0)
        raw_data = np.insert(raw_data, idx, theta)

    for theta in data1:
        idx = np.searchsorted(raw_data, theta)
        data = np.insert(data, idx, [theta, 1], axis = 0)
        raw_data = np.insert(raw_data, idx, theta)

    num_1s = 0
    c_est = data[0][0]
    for (theta, label) in data:
        if label == 1:
            num_1s += 1
        else:
            num_1s -= 1
            if num_1s <= 0:
                c_est = theta
                num_1s = 0

    def emp_risk_test(theta):
        err_count = np.searchsorted(data1, theta, side="right")
        err_count += len(data0) - np.searchsorted(data0, theta, side="left")
        return err_count
        #return len([x for x in data1_test if x <= theta]) + len([x for x in data0_test if x > theta])

    def T(theta, omega):
        return (omega * (emp_risk_test(c_est) - emp_risk_test(theta)))

    # Check that confidence set contains c
    if T(c, omega) >= np.log(1 - nom_coverage):
        universal_coverage += 1

    return (exact_coverage, universal_coverage)



nom_coverages = np.linspace(0, 1, num=100)[90:-1]
exact_coverages = []
universal_coverages = []
for nom_coverage in nom_coverages:
    print("Nominal coverage:", round(nom_coverage, 2))
    mc_iters = 100
    output = [mc_iteration(nom_coverage) for _ in range(mc_iters)]

    exact_coverages.append(np.mean([ec for (ec, uc) in output]))
    universal_coverages.append(np.mean([uc for (ec, uc) in output]))
    print(exact_coverages, universal_coverages)

plt.title("Normally Distributed Data")
plt.scatter(nom_coverages, exact_coverages, color="blue", label = "Exact CI")
plt.scatter(nom_coverages, universal_coverages, color = "red", label = "Universal CI")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.legend()
plt.show()

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

np.random.seed(0)

# 1 dimensional
mu0 = 5
mu1 = 10
cov = 10000
c = (mu0 + mu1)/2


omega = 0.04

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
    omegas = np.linspace(0, 1, num = 10)[1:]
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
    num_tries = 0
    while True:
        data0 = st.norm.rvs(loc = mu0, scale = cov**0.5, size = 100)
        data1 = st.norm.rvs(loc = mu1, scale = cov**0.5, size = 100)
        n0 = 100
        n1 = 100
        # Point estimate for (mu0 + mu1)/2
        c_est = (np.mean(data0) + np.mean(data1))/2

        # (1-\alpha)100% confidence interval for (mu0 + mu1)/2
        z_alpha = st.norm.interval(nom_coverage)[1]
        CI = [c_est - z_alpha* ((n0 + n1)/(4*n0*n1) * cov)**0.5, c_est + z_alpha * ((n0 + n1)/(4*n0*n1) * cov)**0.5]

        # If the estimator is positive, try again
        if  c_est > -10:
            num_tries += 1
            continue

        # Otherwise, we stop collecting data and check if the confidence interval contains the true (mu0+mu1)/2
        if CI[0] <= c and c <= CI[1]:
            exact_coverage += 1
        break

    # Universal interval

    # ERM computation
    universal_coverage = gibbs(c, data0, data1, nom_coverage, omega)

    return (exact_coverage, universal_coverage, num_tries)



nom_coverages = np.linspace(0, 1, num=100)[1:-1]
exact_coverages = []
universal_coverages = []
for nom_coverage in nom_coverages:
    break
    print("Nominal coverage:", round(nom_coverage, 2))
    mc_iters = 1000
    output = [mc_iteration(nom_coverage) for _ in range(mc_iters)]

    exact_coverages.append(np.mean([ec for (ec, uc, nt) in output]))
    universal_coverages.append(np.mean([uc for (ec, uc, nt) in output]))
    print("    Avg. tries:", np.mean([nt for (ec, uc, nt) in output]))
    print("    Coverages:", exact_coverages[-1], universal_coverages[-1])
exact_coverages = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.246]
universal_coverages = [0.509, 0.516, 0.51, 0.49, 0.52, 0.522, 0.517, 0.548, 0.521, 0.527, 0.536, 0.555, 0.568, 0.566, 0.578, 0.605, 0.521, 0.597, 0.585, 0.571, 0.579, 0.62, 0.584, 0.633, 0.594, 0.613, 0.61, 0.623, 0.614, 0.644, 0.656, 0.644, 0.613, 0.645, 0.621, 0.644, 0.637, 0.666, 0.686, 0.678, 0.649, 0.674, 0.679, 0.68, 0.674, 0.677, 0.681, 0.704, 0.716, 0.701, 0.71, 0.7, 0.735, 0.721, 0.701, 0.717, 0.723, 0.706, 0.724, 0.729, 0.738, 0.748, 0.738, 0.767, 0.792, 0.767, 0.784, 0.777, 0.801, 0.749, 0.782, 0.822, 0.802, 0.812, 0.811, 0.835, 0.834, 0.822, 0.822, 0.818, 0.857, 0.863, 0.858, 0.867, 0.884, 0.895, 0.889, 0.915, 0.91, 0.925, 0.909, 0.946, 0.946, 0.961, 0.98, 0.993, 0.999, 1.0]
print(exact_coverages)
print(universal_coverages)

plt.title("Stopping Rule: X̅< -10")
plt.scatter(nom_coverages, exact_coverages, color="blue", label = "Exact CI")
plt.scatter(nom_coverages, universal_coverages, color = "red", label = "Gibbs CS")
plt.plot(nom_coverages, nom_coverages, color = "black")
plt.xlabel("Nominal Coverage")
plt.ylabel("Observed Coverage")
plt.legend()
plt.show()

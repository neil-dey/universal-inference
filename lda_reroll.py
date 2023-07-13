import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

np.random.seed(0)

# 1 dimensional
mu0 = 5
mu1 = 10
cov = 10000
c = (mu0 + mu1)/2


def _gibbs(mu, data0, data1, nom_coverage, omega):
    data0_train = data0[:len(data0)//2]
    data0_test = sorted(data0[len(data0)//2:])
    data1_train = data1[:len(data1)//2]
    data1_test = sorted(data1[len(data1)//2:])

    raw_data = np.array([])
    train_data = np.empty(shape=(0, 2))
    for theta in data0_train:
        idx = np.searchsorted(raw_data, theta)
        train_data = np.insert(train_data, idx, [theta, 0], axis = 0)
        raw_data = np.insert(raw_data, idx, theta)

    for theta in data1_train:
        idx = np.searchsorted(raw_data, theta)
        train_data = np.insert(train_data, idx, [theta, 1], axis = 0)
        raw_data = np.insert(raw_data, idx, theta)

    num_1s = 0
    c_est = train_data[0][0]
    for (theta, label) in train_data:
        if label == 1:
            num_1s += 1
        else:
            num_1s -= 1
            if num_1s <= 0:
                c_est = theta
                num_1s = 0

    def emp_risk_test(theta):
        err_count = np.searchsorted(data1_test, theta, side="right")
        err_count += len(data0_test) - np.searchsorted(data0_test, theta, side="left")
        return err_count
        #return len([x for x in data1_test if x <= theta]) + len([x for x in data0_test if x > theta])

    def T(theta, omega):
        return (omega * (emp_risk_test(c_est) - emp_risk_test(theta)))

    # Check that confidence set contains mu
    return T(mu, omega) >= np.log(1 - nom_coverage)

def gibbs(mu, data0, data1, nom_coverage):
    boot_iters = 100
    coverages = []
    omegas = np.linspace(0.05, 1, num = 20)
    for omega in omegas:
        coverage = 0
        for _ in range(boot_iters):
            boot_data0 = np.random.choice(data0, size = len(data0), replace = True)
            boot_data1 = np.random.choice(data1, size = len(data1), replace = True)
            boot_c = (np.mean(data0) + np.mean(data1))/2
            coverage += _gibbs(boot_c, boot_data0, boot_data1, nom_coverage, omega)
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

        # If the estimator is large, try again
        if  c_est > -10:
            num_tries += 1
            continue

        # (1-\alpha)100% confidence interval for (mu0 + mu1)/2
        z_alpha = st.norm.interval(nom_coverage)[1]
        CI = [c_est - z_alpha* ((n0 + n1)/(4*n0*n1) * cov)**0.5, c_est + z_alpha * ((n0 + n1)/(4*n0*n1) * cov)**0.5]
        # Otherwise, we stop collecting data and check if the confidence interval contains the true (mu0+mu1)/2
        if CI[0] <= c and c <= CI[1]:
            exact_coverage += 1
        break

    # Universal interval
    universal_coverage = gibbs(c, data0, data1, nom_coverage)

    return (exact_coverage, universal_coverage, num_tries)



nom_coverages = np.linspace(0, 1, num=100)[80:-1]
exact_coverages = []
universal_coverages = []
for nom_coverage in nom_coverages:
    print("Nominal coverage:", round(nom_coverage, 2))
    mc_iters = 100
    output = [mc_iteration(nom_coverage) for _ in range(mc_iters)]

    exact_coverages.append(np.mean([ec for (ec, uc, nt) in output]))
    universal_coverages.append(np.mean([uc for (ec, uc, nt) in output]))
    print("    Avg. tries:", np.mean([nt for (ec, uc, nt) in output]))
    print("    Coverages:", exact_coverages[-1], universal_coverages[-1])
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
